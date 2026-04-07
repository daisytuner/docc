#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Computes closed-form final value of induction variable after loop
 *
 * For a normalized loop (init=0, stride=1), sets indvar = num_iterations after the loop.
 *
 * This breaks any write-after-read dependency on the induction variable
 * in subsequent blocks. SymbolPropagation will then propagate this value
 * into downstream assignments, and dead code elimination will clean up.
 *
 * Prerequisites:
 * - Loop must be in normal form (init=0, stride=1, canonical bound extractable)
 * - Must know the original init and stride before normalization
 */
class LoopIndvarFinalize : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    /**
     * @brief Construct transform to finalize indvar value after normalization
     * @param loop The normalized loop
     */
    LoopIndvarFinalize(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopIndvarFinalize from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
