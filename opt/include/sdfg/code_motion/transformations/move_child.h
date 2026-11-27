#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace code_motion {
namespace transformations {

/**
 * @brief Moves a child of a sequence to another position in it
 *
 * The child runs before or after the siblings between its position and the target instead. Legal if the child and
 * every sibling it passes are movable (no return, break, continue or library node with side effects) and the child
 * conflicts with none of them (see code_motion::may_conflict): for each container one of them writes, the elements
 * they access are disjoint.
 */
class MoveChild : public ::sdfg::transformations::Transformation {
    structured_control_flow::ControlFlowNode& child_;
    size_t position_;

public:
    /**
     * @param child A child of a sequence
     * @param position The index the child takes in the sequence
     */
    MoveChild(structured_control_flow::ControlFlowNode& child, size_t position);

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static MoveChild from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace code_motion
} // namespace sdfg
