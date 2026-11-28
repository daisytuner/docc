#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace code_motion {
namespace transformations {

/**
 * @brief Moves a child of a loop's body before or after the loop
 *
 * The child runs once instead of in every iteration. Legal if
 * - the loop provably runs at least once,
 * - the child does not use the loop's counter,
 * - the child reads no container it writes, so that running it again computes the same,
 * - the child and the body's other children are movable (no return, break, continue or library node with side
 *   effects),
 * - no other child of the body writes what the child reads or writes, in any pair of iterations, and
 * - no other child reads what it writes, except, moving before the loop, those after it in the body: every iteration
 *   still finds it written (see code_motion::may_conflict).
 */
class HoistChild : public ::sdfg::transformations::Transformation {
    structured_control_flow::StructuredLoop& loop_;
    structured_control_flow::ControlFlowNode& child_;
    bool after_;

public:
    /**
     * @param loop The loop
     * @param child A child of the loop's body
     * @param after Whether the child moves after the loop rather than before
     */
    HoistChild(
        structured_control_flow::StructuredLoop& loop,
        structured_control_flow::ControlFlowNode& child,
        bool after = false
    );

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static HoistChild from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace code_motion
} // namespace sdfg
