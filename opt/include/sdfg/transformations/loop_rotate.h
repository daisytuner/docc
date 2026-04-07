#pragma once

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop rotation transformation to convert negative stride to positive stride
 *
 * This transformation reverses the iteration direction of a loop with negative stride.
 *
 * Original loop (negative stride):
 *   for (i = init; bound < i; i -= 1)   // or equivalently: i > bound
 *       body(i)
 *
 * After LoopRotate:
 *   for (i = bound+1; i < init+1; i += 1)
 *       body(init + bound + 1 - i)     // Transformed index
 *
 * Example:
 *   // Original: i = 10, 9, 8, ..., 1 (while i > 0)
 *   for (i = 10; 0 < i; i--)
 *       A[i] = ...
 *
 *   // After rotation: i' = 1, 2, 3, ..., 10
 *   for (i = 1; i < 11; i++)
 *       A[11 - i] = ...   // 11-1=10, 11-2=9, ..., 11-10=1
 *
 * Prerequisites:
 * - Loop must have stride == -1
 * - Loop condition must have extractable lower bound (form: bound < indvar)
 *
 * Benefits:
 * - Enables further transformations that require positive stride
 * - Normalizes loops for analysis
 * - Works with both For and Map loops
 *
 * @note After applying this transformation, you should run SymbolPropagation
 *       to propagate the rotated index expression into the loop body.
 */
class LoopRotate : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

    std::string rotated_container_name_;

public:
    /**
     * @brief Construct a LoopRotate transformation
     * @param loop The loop to rotate (must have negative stride)
     */
    explicit LoopRotate(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopRotate from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /**
     * @brief Get the name of the container holding the rotated (original) value
     * @return The container name, or empty string if not yet applied
     */
    const std::string& rotated_container_name() const { return rotated_container_name_; }
};

} // namespace transformations
} // namespace sdfg
