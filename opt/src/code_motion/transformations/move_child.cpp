#include "sdfg/code_motion/transformations/move_child.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/code_motion/conflicts.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/sequence.h"

namespace sdfg {
namespace code_motion {
namespace transformations {

MoveChild::MoveChild(structured_control_flow::ControlFlowNode& child, size_t position)
    : child_(child), position_(position) {
}

std::string MoveChild::name() const {
    return "MoveChild";
}

bool MoveChild::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto* parent = dynamic_cast<structured_control_flow::Sequence*>(child_.get_parent());
    if (!parent || position_ >= parent->size()) {
        return false;
    }
    size_t index = parent->index(child_);
    if (index == position_) {
        return false;
    }
    if (!movable(builder.subject(), child_)) {
        return false;
    }
    size_t first = std::min(index, position_);
    size_t last = std::max(index, position_);
    for (size_t k = first; k <= last; ++k) {
        if (k != index && !movable(builder.subject(), parent->at(k))) {
            return false;
        }
    }
    auto& scope = analysis_manager.get<analysis::AssumptionsAnalysis>().get(*parent, true);
    auto child = accesses_of(builder.subject(), analysis_manager, child_);
    for (size_t k = first; k <= last; ++k) {
        if (k != index && may_conflict(child, accesses_of(builder.subject(), analysis_manager, parent->at(k)), scope)) {
            return false;
        }
    }
    return true;
}

void MoveChild::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& parent = static_cast<structured_control_flow::Sequence&>(*child_.get_parent());
    builder.move_child(parent, parent.index(child_), parent, position_);
    // Only the order within the sequence changed.
    analysis_manager.invalidate_all(parent);
}

void MoveChild::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = {{"position", position_}};
    serializer::JSONSerializer serializer(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["0"], child_);
}

MoveChild MoveChild::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto id = j["subgraph"]["0"]["element_id"].get<size_t>();
    auto* child = dynamic_cast<structured_control_flow::ControlFlowNode*>(builder.find_element_by_id(id));
    if (child == nullptr) {
        throw ::sdfg::transformations::InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(id) + " is not a control-flow node."
        );
    }
    return MoveChild(*child, j["parameters"]["position"].get<size_t>());
}

} // namespace transformations
} // namespace code_motion
} // namespace sdfg
