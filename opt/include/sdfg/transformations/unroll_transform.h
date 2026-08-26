#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Marks a constant-trip loop for full unrolling.
 *
 * Sets the `"unroll"` property on the loop's existing schedule (preserving its
 * kind, e.g. SEQUENTIAL / CUDA), which the loop codegen lowers to
 * `#pragma clang loop unroll(full)`. Full unrolling of the constant-trip loops
 * that index a register tile is what lets the compiler scalarize the tile into
 * registers (instead of spilling it to local memory).
 *
 * Only applicable to loops with a provably constant (positive integer) trip
 * count, so `unroll(full)` is never emitted on a variable-trip loop (which the
 * compiler would reject).
 */
class UnrollTransform : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    explicit UnrollTransform(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static UnrollTransform from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
