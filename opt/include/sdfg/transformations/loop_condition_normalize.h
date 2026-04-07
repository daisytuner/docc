#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop condition normalization to convert inequality conditions to relational comparisons
 *
 * This transformation converts loop conditions with `!=` (Unequality) or `==` (Equality)
 * to relational comparisons (`<` or `>`) based on the loop's stride direction.
 *
 * Precondition: stride is ±1 (apply LoopUnitStride first if needed)
 *
 * For positive stride (+1):
 *   i != N  →  i < N    (loop runs while i < N, exits when reaching N)
 *
 * For negative stride (-1):
 *   i != N  →  i > N    (loop runs while i > N, exits when reaching N)
 *
 * Example (LLVM-style loop with != condition):
 *   // Original: for (i = 0; i != 10; i++)
 *   for (i = 0; i != 10; i += 1)
 *       body(i)
 *
 *   // After normalization:
 *   for (i = 0; i < 10; i += 1)
 *       body(i)
 *
 * The transformation handles affine expressions in the condition:
 *   i != 2*N + 1  →  i < 2*N + 1    (for stride = +1)
 *
 * Prerequisites:
 * - Loop must have unit stride (|stride| == 1)
 * - Condition must contain at least one Unequality with the induction variable
 * - The inequality must be normalizable to form: indvar != bound
 *
 * Non-applicability:
 * - Stride is not ±1
 * - Condition has no Unequality/Equality literals
 * - indvar appears on both sides of the condition
 * - Condition is not affine in indvar
 *
 * @note This transformation is typically applied after LoopUnitStride and before LoopRotate
 *       in the loop normalization pipeline.
 *
 * @see LoopShift
 * @see LoopUnitStride
 * @see LoopRotate
 */
class LoopConditionNormalize : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    /**
     * @brief Construct a LoopConditionNormalize transformation
     * @param loop The loop to normalize (must have unit stride)
     */
    explicit LoopConditionNormalize(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopConditionNormalize from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
