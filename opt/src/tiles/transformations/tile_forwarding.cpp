#include "sdfg/tiles/transformations/tile_forwarding.h"

#include <optional>
#include <string>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/code_motion/conflicts.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"

namespace sdfg {
namespace transformations {

namespace {

/// A block holding one tile copy between a local buffer and a global container.
struct StagingCopy {
    const tiles::TileCopyNode* node;
    std::string global;
    std::string local;
};

std::optional<StagingCopy> staging_copy(structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    if (dataflow.library_nodes().size() != 1 || dataflow.nodes().size() != 3) {
        return std::nullopt;
    }
    auto* copy = dynamic_cast<const tiles::TileCopyNode*>(*dataflow.library_nodes().begin());
    if (!copy) {
        return std::nullopt;
    }
    // Copying in, `_src` is the global container; copying out, `_dst` is.
    bool in = copy->direction() == tiles::CopyDirection::In;
    StagingCopy result{copy, "", ""};
    for (auto& memlet : dataflow.in_edges(*copy)) {
        auto* access = dynamic_cast<const data_flow::AccessNode*>(&memlet.src());
        if (!access || dynamic_cast<const data_flow::ConstantNode*>(access)) {
            return std::nullopt;
        }
        ((memlet.dst_conn() == "_src") == in ? result.global : result.local) = access->data();
    }
    if (result.global.empty() || result.local.empty() || result.global == result.local) {
        return std::nullopt;
    }
    return result;
}

/// Whether @p a and @p b address the same elements, whatever form their expressions take.
bool same_layout(const math::tensor::TensorLayout& a, const math::tensor::TensorLayout& b) {
    auto equal = [](const symbolic::Expression& x, const symbolic::Expression& y) {
        return symbolic::eq(symbolic::expand(x), symbolic::expand(y));
    };
    if (a.dims() != b.dims() || !equal(a.offset(), b.offset())) {
        return false;
    }
    for (int d = 0; d < a.dims(); ++d) {
        if (!equal(a.get_dim(d), b.get_dim(d)) || !equal(a.get_stride(d), b.get_stride(d))) {
            return false;
        }
    }
    return true;
}

/// Whether @p copy moves a whole tile, each element once.
bool whole(const tiles::TileCopyNode& copy) {
    return copy.phase() == tiles::CopyPhase::Full && copy.plan().dst_swizzle.is_identity() && copy.guard().trivial();
}

} // namespace

TileForwarding::TileForwarding(structured_control_flow::Block& earlier, structured_control_flow::Block& later)
    : earlier_(earlier), later_(later) {
}

std::string TileForwarding::name() const {
    return "TileForwarding";
}

bool TileForwarding::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(earlier_.get_parent());
    if (!sequence || later_.get_parent() != sequence) {
        return false;
    }
    size_t first = sequence->index(earlier_);
    size_t last = sequence->index(later_);
    if (first >= last) {
        return false;
    }
    auto earlier = staging_copy(earlier_);
    auto later = staging_copy(later_);
    if (!earlier || !later || later->node->direction() != tiles::CopyDirection::In ||
        earlier->global != later->global || earlier->local == later->local || !whole(*earlier->node) ||
        !whole(*later->node) || !same_layout(earlier->node->plan().src, later->node->plan().src) ||
        !same_layout(earlier->node->plan().dst, later->node->plan().dst) ||
        !(sdfg.type(earlier->local) == sdfg.type(later->local))) {
        return false;
    }

    // Nothing between the copies writes the tile, the earlier buffer or a symbol of the earlier plan.
    auto& scope = analysis_manager.get<analysis::AssumptionsAnalysis>().get(*sequence, true);
    auto copied = code_motion::accesses_of(sdfg, analysis_manager, earlier_);
    for (size_t k = first + 1; k < last; ++k) {
        auto& step = sequence->at(k);
        if (!code_motion::movable(sdfg, step)) {
            return false;
        }
        code_motion::Dependences writes{.flow = false, .anti = true, .output = true};
        if (code_motion::may_conflict(copied, code_motion::accesses_of(sdfg, analysis_manager, step), scope, writes)) {
            return false;
        }
    }

    // The later buffer lives in the sequence from the later copy on, and neither buffer changes there.
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView in_sequence(users, *sequence);
    if (users.uses(later->local).size() != in_sequence.uses(later->local).size()) {
        return false;
    }
    for (size_t k = 0; k < sequence->size(); ++k) {
        analysis::UsersView step(users, sequence->at(k));
        if (k < last && !step.uses(later->local).empty()) {
            return false;
        }
        if (k > last) {
            for (auto& buffer : {earlier->local, later->local}) {
                if (!step.writes(buffer).empty() || !step.moves(buffer).empty() || !step.views(buffer).empty()) {
                    return false;
                }
            }
        }
    }
    return true;
}

void TileForwarding::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sequence = static_cast<structured_control_flow::Sequence&>(*later_.get_parent());
    auto earlier = staging_copy(earlier_);
    auto later = staging_copy(later_);
    auto local = later->local;
    builder.remove_child(sequence, sequence.index(later_));
    sequence.replace(symbolic::symbol(local), symbolic::symbol(earlier->local));
    if (builder.subject().is_transient(local)) {
        builder.remove_container(local);
    }
    analysis_manager.invalidate_all();
}

void TileForwarding::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    serializer::JSONSerializer serializer(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["0"], earlier_);
    j["subgraph"]["1"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["1"], later_);
}

TileForwarding TileForwarding::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto earlier_id = j["subgraph"]["0"]["element_id"].get<size_t>();
    auto later_id = j["subgraph"]["1"]["element_id"].get<size_t>();
    auto* earlier = dynamic_cast<structured_control_flow::Block*>(builder.find_element_by_id(earlier_id));
    auto* later = dynamic_cast<structured_control_flow::Block*>(builder.find_element_by_id(later_id));
    if (earlier == nullptr || later == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Elements with IDs " + std::to_string(earlier_id) + " and " + std::to_string(later_id) + " are no blocks."
        );
    }
    return TileForwarding(*earlier, *later);
}

} // namespace transformations
} // namespace sdfg
