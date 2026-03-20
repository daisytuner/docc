#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop interchange transformation that swaps two nested loops
 *
 * This transformation swaps the order of two nested loops (outer and inner),
 * potentially improving data locality and parallelization opportunities.
 * The transformation preserves the computational semantics while reordering
 * loop execution.
 *
 * Supports two cases:
 * - Independent bounds: inner loop bounds do not reference the outer induction
 *   variable. The loop headers are simply swapped.
 * - Dependent bounds (Fourier-Motzkin): inner loop bounds are unit-affine in
 *   the outer induction variable. The new outer range is obtained by projecting
 *   the inner bounds over the outer range, and the new inner range is obtained
 *   by inverting the affine relationship, yielding min/max expressions.
 *
 * @note The outer loop must have exactly one child (the inner loop)
 * @note For-For interchange is checked via dependence analysis (delta sets must remain lex-non-negative)
 */
class LoopInterchange : public Transformation {
    structured_control_flow::StructuredLoop& outer_loop_;
    structured_control_flow::StructuredLoop& inner_loop_;
    bool applied_ = false;
    structured_control_flow::StructuredLoop* new_outer_loop_;
    structured_control_flow::StructuredLoop* new_inner_loop_;

public:
    /**
     * @brief Construct a loop interchange transformation
     * @param outer_loop The outer loop to be interchanged
     * @param inner_loop The inner loop to be interchanged
     */
    LoopInterchange(
        structured_control_flow::StructuredLoop& outer_loop, structured_control_flow::StructuredLoop& inner_loop
    );

    /**
     * @brief Get the name of this transformation
     * @return "LoopInterchange"
     */
    virtual std::string name() const override;

    /**
     * @brief Check if this transformation can be applied
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     * @return true if the transformation can be applied safely
     */
    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    /**
     * @brief Apply the loop interchange transformation
     * @param builder The SDFG builder
     * @param analysis_manager The analysis manager
     */
    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    /**
     * @brief Serialize this transformation to JSON
     * @param j JSON object to populate
     */
    virtual void to_json(nlohmann::json& j) const override;

    /**
     * @brief Deserialize a loop interchange transformation from JSON
     * @param builder The SDFG builder
     * @param j JSON description of the transformation
     * @return The deserialized transformation
     */
    static LoopInterchange from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    structured_control_flow::StructuredLoop* new_outer_loop() const;
    structured_control_flow::StructuredLoop* new_inner_loop() const;
};

} // namespace transformations
} // namespace sdfg
