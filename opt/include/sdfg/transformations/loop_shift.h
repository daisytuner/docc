#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop shift transformation to change loop initialization value
 *
 * This transformation shifts a loop's iteration space.
 *
 * Original loop:
 *   for (i = init; i < bound; i += stride)
 *       body(i)
 *
 * After LoopShift(target_init=0):
 *   for (i = 0; i < bound - init; i += stride)
 *       { __i_orig = i + init; }  // Assignment in transition
 *       body(__i_orig)            // Uses original value via propagation
 *
 * Usage:
 *   1. Apply LoopShift transformation
 *   2. Run SymbolPropagation pass to propagate the assignment
 *   3. Run DeadCodeElimination to remove the temporary container if unused
 *
 * Prerequisites:
 * - Loop must have an integer init value (for condition adjustment)
 * - Loop body must not be empty
 *
 * @note After applying this transformation, you should run SymbolPropagation
 *       to propagate the shifted value into the loop body.
 */
class LoopShift : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    symbolic::Expression offset_;

    std::string shifted_container_name_;

public:
    /**
     * @brief Construct a LoopShift transformation to shift loop to start from 0
     * @param loop The loop to shift
     */
    explicit LoopShift(structured_control_flow::StructuredLoop& loop);

    /**
     * @brief Construct a LoopShift transformation with a custom offset
     * @param loop The loop to shift
     * @param offset The offset to apply to the loop's initial value (e.g., if offset = 1, then new_init = old_init - 1)
     */
    LoopShift(structured_control_flow::StructuredLoop& loop, const symbolic::Expression& offset);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopShift from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the name of the container holding the shifted (original) value
     * @return The container name, or empty string if not yet applied
     */
    const std::string& shifted_container_name() const { return shifted_container_name_; }
};

} // namespace transformations
} // namespace sdfg
