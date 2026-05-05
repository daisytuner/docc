#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop splitting transformation at a symbolic boundary
 *
 * Splits a single loop into two consecutive loops at the same nesting level,
 * partitioning the iteration space at a symbolic split point.
 *
 * Given:
 *   for (i = init; i < bound; i = update) { body }
 *
 * Produces:
 *   for (i = init; i < split_point; i = update) { body_copy }
 *   for (i = split_point; i < bound; i = update) { body }
 *
 * The split point is an arbitrary symbolic expression (e.g., an enclosing
 * loop's induction variable). The transformation is always semantically
 * correct — it is an identity on the iteration space:
 *   [init, bound) = [init, split_point) ∪ [split_point, bound)
 *
 * If split_point <= init, the first loop is empty (zero iterations).
 * If split_point >= bound, the second loop is empty (zero iterations).
 *
 * @note The loop must be contiguous (stride == 1)
 * @note The loop condition must be a simple strict-less-than (indvar < bound)
 * @note The split_point must be a valid symbolic expression in scope
 *
 * Primary use case: enabling LoopDistribute on triangular loop nests
 * (e.g., splitting k in [0, i) at i_tile0 for blocked LU factorization).
 */
class LoopSplit : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    symbolic::Expression split_point_;

public:
    /**
     * @brief Construct a loop split transformation
     * @param loop The loop to split
     * @param split_point Symbolic expression at which to split the iteration space
     */
    LoopSplit(structured_control_flow::StructuredLoop& loop, const symbolic::Expression& split_point);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopSplit from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
