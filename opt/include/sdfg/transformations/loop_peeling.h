#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop peeling transformation for compound-condition loops
 *
 * This transformation splits a loop with a compound condition (conjunction)
 * into a main body with a simple constant-trip-count condition and a postamble
 * for the remainder case. The result is an IfElse node:
 *
 *   if (canonical_bound <= min(dynamic_bounds)):
 *     loop(init, canonical_bound, step) { body }        // constant trip count
 *   else:
 *     loop(init, min(dynamic_bounds), step) { body }    // remainder
 *
 * The canonical bound is the conjunct that gives a constant trip count relative
 * to the init expression (e.g., `j < 8 + j_tile1` with init=j_tile1 → trip=8).
 *
 * This enables the compiler to vectorize the main body with a proven trip count.
 */
class LoopPeeling : public Transformation {
    structured_control_flow::StructuredLoop& loop_;

public:
    /**
     * @brief Construct a loop peeling transformation
     * @param loop The loop with compound conditions to be peeled
     */
    LoopPeeling(structured_control_flow::StructuredLoop& loop);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopPeeling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
