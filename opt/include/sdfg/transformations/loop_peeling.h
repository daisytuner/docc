#pragma once

#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Loop peeling for a perfectly nested chain of compound-condition loops.
 *
 * Targets a loop whose condition is a conjunction of a canonical (constant-trip)
 * bound and one or more dynamic bounds, e.g. `for (k = k0; k < TK + k0 && k < N; ++k)`,
 * and greedily collects the perfectly nested chain of such loops beneath it. All
 * loops in the chain are over-approximated to their constant trip counts and
 * shifted to a 0-based induction variable, so the whole nest becomes
 * compile-time constant and can be fully unrolled/vectorized. The dropped dynamic
 * bounds are re-applied in one of two ways, selected by @p predicate:
 *
 * - **Hoisted (default, `predicate = false`)** — a single outer `IfElse` whose
 *   "then" branch runs the clean, unguarded, 0-based nest when the whole tile is
 *   in bounds, and whose "else" branch runs the original (variable-trip) nest for
 *   boundary tiles. The unguarded inner micro-kernel vectorizes on CPU. Universal.
 *
 * - **Predicated (`predicate = true`)** — the 0-based nest is emitted directly and
 *   the innermost body is wrapped in one combined guard `if (all dynamic bounds
 *   hold)`, with no remainder branch. On GPU the guard lowers to cheap predicated
 *   instructions, so register-tile accumulators stay in registers (no local-memory
 *   spill). Use for GPU / einsum register tiling.
 */
class LoopPeeling : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    bool predicate_;

public:
    /**
     * @brief Construct a loop peeling transformation.
     * @param loop The outermost loop of the perfectly nested compound-condition chain.
     * @param predicate If true, emit the predicated (GPU register-tiling) form; if
     *        false (default), emit the hoisted then/else form.
     */
    explicit LoopPeeling(structured_control_flow::StructuredLoop& loop, bool predicate = false);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopPeeling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
