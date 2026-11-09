#include "sdfg/tiles/transformations/tile_streaming_copy.h"

#include <functional>
#include <limits>
#include <set>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/users.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_reduce_layout.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"

namespace sdfg {
namespace transformations {

namespace {

using structured_control_flow::ControlFlowNode;

// Each source element of @p node is read once per kernel launch: the source footprint, folded with
// every enclosing in-kernel loop (innermost first), is an injective affine layout over a
// launch-uniform origin, and every launched thread executes it for exactly one iteration of the
// enclosing GPU loops. Caching such a load only evicts reused data.
bool reads_source_once(
    tiles::TileCopyNode& node, const Function& function, analysis::AnalysisManager& analysis_manager
) {
    const auto& src = node.plan().src;
    std::vector<gpu::ReductionLayout::Dimension> axes;
    for (size_t d = 0; d < src.shape().size(); ++d) {
        const auto* count = dynamic_cast<const SymEngine::Integer*>(src.shape()[d].get());
        const auto* stride = dynamic_cast<const SymEngine::Integer*>(src.strides()[d].get());
        if (count == nullptr || stride == nullptr) {
            return false;
        }
        axes.push_back({stride->as_int(), count->as_int()});
    }
    // Enclosing control flow, innermost first.
    std::vector<ControlFlowNode*> enclosing;
    for (auto* n = dynamic_cast<ControlFlowNode*>(node.get_parent().get_parent()); n != nullptr;
         n = n->get_parent()) {
        enclosing.push_back(n);
    }
    std::vector<structured_control_flow::StructuredLoop*> loops;
    // loops[0, kernel_depth) run inside the kernel.
    size_t kernel_depth = 0;
    bool while_in_kernel = false;
    bool while_seen = false;
    for (auto* n : enclosing) {
        if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(n)) {
            loops.push_back(loop);
            if (loop->schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader) {
                kernel_depth = loops.size();
                while_in_kernel = while_seen;
            }
        } else if (dynamic_cast<structured_control_flow::While*>(n) != nullptr) {
            while_seen = true;
        }
    }
    // A While inside the kernel repeats the copy with no provable footprint movement.
    if (kernel_depth == 0 || while_in_kernel) {
        return false;
    }

    // Each thread must own a distinct iteration of the enclosing GPU loops. Lanes and WARP loops
    // share an iteration across threads, and a launched level without an enclosing loop runs the
    // copy on every thread of that level.
    std::set<gpu::TargetLevel> enclosing_levels;
    for (size_t i = 0; i < kernel_depth; ++i) {
        const auto& schedule = loops[i]->schedule_type();
        if (schedule.category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        auto level = gpu::ScheduleType_GPU_Offload::target_level_if_exists(schedule);
        if (!level || *level == gpu::TargetLevel::WARP ||
            gpu::ScheduleType_GPU_Offload::lanes(schedule)->as_int() != 1) {
            return false;
        }
        enclosing_levels.insert(*level);
    }
    std::unordered_map<gpu::TargetLevel, structured_control_flow::ScheduleType> launch;
    gpu::get_nested_schedule_types(*loops[kernel_depth - 1], analysis_manager, launch);
    for (const auto& [level, schedule] : launch) {
        if (level == gpu::TargetLevel::WARP) {
            return false;
        }
        if (enclosing_levels.count(level) == 0 && gpu::ScheduleType_GPU_Offload::threads(schedule)->as_int() > 1) {
            return false;
        }
    }
    auto base = src.offset();
    for (size_t i = 0; i < kernel_depth; ++i) {
        const auto* loop = loops[i];
        auto count = loop->num_iterations();
        // Iterations that do not move the footprint read it again.
        if (!symbolic::uses(base, loop->indvar())) {
            if (count.is_null() || !symbolic::eq(count, symbolic::one())) {
                return false;
            }
            continue;
        }
        auto origin = symbolic::subs(base, loop->indvar(), symbolic::zero());
        auto coefficient =
            symbolic::expand(symbolic::sub(symbolic::subs(base, loop->indvar(), symbolic::one()), origin));
        if (count.is_null() || !SymEngine::is_a<SymEngine::Integer>(*count) ||
            !SymEngine::is_a<SymEngine::Integer>(*coefficient) ||
            !symbolic::
                eq(symbolic::expand(base),
                   symbolic::expand(symbolic::add(origin, symbolic::mul(coefficient, loop->indvar()))))) {
            return false;
        }
        const auto count_value = SymEngine::rcp_static_cast<const SymEngine::Integer>(count)->as_int();
        const auto coefficient_value = SymEngine::rcp_static_cast<const SymEngine::Integer>(coefficient)->as_int();
        const auto stride_value = loop->stride()->as_int();
        if (count_value <= 0 || coefficient_value <= 0 || stride_value <= 0 ||
            coefficient_value > std::numeric_limits<int64_t>::max() / stride_value) {
            return false;
        }
        axes.push_back({coefficient_value * stride_value, count_value});
        base = symbolic::subs(base, loop->indvar(), loop->init());
    }
    // The origin must be fixed for the launch: arguments or indvars of host loops around the kernel,
    // none of them reassigned inside the kernel.
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView kernel_users(users, *loops[kernel_depth - 1]);
    for (const auto& atom : symbolic::atoms(base)) {
        const auto& name = atom->get_name();
        bool uniform = function.is_argument(name);
        for (size_t i = kernel_depth; i < loops.size() && !uniform; ++i) {
            uniform = symbolic::eq(atom, loops[i]->indvar());
        }
        if (!uniform || !kernel_users.writes(name).empty() || !kernel_users.moves(name).empty() ||
            !kernel_users.views(name).empty()) {
            return false;
        }
    }
    try {
        gpu::ReductionLayout(base, std::move(axes));
    } catch (const InvalidSDFGException&) {
        return false;
    }
    return true;
}

// The not-yet-streaming per-thread vector copy-ins under @p root that read their source once.
std::vector<tiles::TileCopyNode*> streamable_copies(
    ControlFlowNode& root, const Function& function, analysis::AnalysisManager& analysis_manager
) {
    std::vector<tiles::TileCopyNode*> copies;
    std::function<void(ControlFlowNode&)> walk = [&](ControlFlowNode& n) {
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&n)) {
            for (auto& node : block->dataflow().nodes()) {
                auto* tc = dynamic_cast<tiles::TileCopyNode*>(&node);
                if (tc != nullptr && !tc->streaming() && tc->direction() == tiles::CopyDirection::In &&
                    tc->atom() == tiles::CopyAtom::VectorSync &&
                    tc->implementation_type() == data_flow::ImplementationType_NONE &&
                    reads_source_once(*tc, function, analysis_manager)) {
                    copies.push_back(tc);
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); i++) {
                walk(seq->at(i));
            }
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            walk(loop->root());
        } else if (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ie->size(); i++) {
                walk(ie->at(i).first);
            }
        }
    };
    walk(root);
    return copies;
}

} // namespace

TileStreamingCopy::TileStreamingCopy(structured_control_flow::StructuredLoop& loop) : loop_(loop) {
}

std::string TileStreamingCopy::name() const {
    return "TileStreamingCopy";
}

bool TileStreamingCopy::can_be_applied(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
) {
    return !streamable_copies(loop_.root(), builder.subject(), analysis_manager).empty();
}

void TileStreamingCopy::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    for (auto* tc : streamable_copies(loop_.root(), builder.subject(), analysis_manager)) {
        tc->set_streaming(true);
    }
    analysis_manager.invalidate_all();
}

void TileStreamingCopy::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
}

TileStreamingCopy TileStreamingCopy::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j["subgraph"]["0"]["element_id"].get<size_t>();
    auto* element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a structured loop."
        );
    }
    return TileStreamingCopy(*loop);
}

} // namespace transformations
} // namespace sdfg
