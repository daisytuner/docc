#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop unit stride transformation to convert non-unit stride loops to unit stride
 *
 * This transformation converts a loop with stride != 1 to a loop with |stride| = 1
 * using symbolic substitution on the condition.
 *
 * Precondition: init == 0 (apply LoopShift first if needed)
 *
 * Original loop (positive stride s > 1):
 *   for (i = 0; i < bound; i += s)
 *       body(i)
 *
 * After LoopUnitStride:
 *   for (i = 0; s*i < bound; i++)
 *       __i_orig__ = s * i          // Assignment in transition
 *       body(__i_orig__)            // Uses original value via propagation
 *
 * For negative stride (s < -1):
 *   for (i = 0; bound < i; i += s)  // s is negative
 *       body(i)
 *
 * After LoopUnitStride:
 *   for (i = 0; bound < s*i; i++)
 *       __i_orig__ = s * i          // s is negative, so this decreases
 *       body(__i_orig__)
 *
 * Example:
 *   // Original: i = 0, 2, 4, 6 (stride = 2, bound = 8)
 *   for (i = 0; i < 8; i += 2)
 *       A[i] = ...
 *
 *   // After: condition becomes 2*i < 8
 *   for (i = 0; 2*i < 8; i++)
 *       __i_orig__ = 2 * i   // = 0, 2, 4, 6
 *       A[__i_orig__] = ...
 *
 * Prerequisites:
 * - Loop init must be 0 (apply LoopShift first)
 * - Loop must have detectable non-unit stride (stride != 1 and stride != -1)
 *
 * Benefits:
 * - Normalizes loop stride for analysis and transformation
 * - Enables TileFusion which requires matching strides
 * - Enables LoopRotate (which requires stride = -1)
 *
 * @note After applying this transformation, you should run SymbolPropagation
 *       to propagate the strided index expression into the loop body.
 *       The condition may need simplification via symbolic simplify pass.
 */
class LoopUnitStride : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

    std::string strided_container_name_;

public:
    /**
     * @brief Construct a LoopUnitStride transformation
     * @param loop The loop to normalize (must have non-unit stride)
     */
    explicit LoopUnitStride(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopUnitStride from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the name of the container holding the strided (original) value
     * @return Container name (e.g., "__i_orig__")
     */
    std::string strided_container_name() const;
};

} // namespace transformations
} // namespace sdfg
