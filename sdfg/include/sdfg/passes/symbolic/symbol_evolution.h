#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Rewrites loop-carried scalar recurrences into closed-form expressions.
 *
 * For every structured loop the pass identifies integer-scalar symbols whose
 * body update can be modelled as the affine recurrence
 *
 *     sym_{n+1} = a * sym_n + b * indvar_n + c     (a, b, c loop-invariant)
 *
 * and rewrites them so the body redefines the symbol from the induction
 * variable directly. This breaks loop-carried dependencies and exposes the
 * symbol as an affine function of the induction variable to downstream
 * polyhedral / memlet analyses.
 *
 * Solvable recurrence classes:
 *   - a == 1, b == 0  (accumulator):     closed(i) = sym_init + c * (i - init) / stride
 *   - a == 0          (function of i):   closed(i) = b * (i - stride) + c
 *
 * Loops are processed innermost-first; each loop is rewritten to a fixpoint,
 * with analyses invalidated between rewrites because each rewrite mutates the
 * IR.
 */
class SymbolEvolution : public Pass {
private:
    /**
     * @brief Attempts a single rewrite within @p loop.
     *
     * Performs cheap structural filtering, then runs the affine recurrence
     * solver, and only then pays for dominance / use-after-update analyses.
     * Returns on the first successful rewrite so the caller can refresh
     * analyses before retrying.
     *
     * @return true if exactly one symbol was rewritten.
     */
    bool eliminate_symbols(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        structured_control_flow::Transition& transition
    );

public:
    /**
     * @brief Constructs a new SymbolEvolution pass instance.
     */
    SymbolEvolution();

    /**
     * @brief Returns the name of this pass.
     *
     * @return "SymbolEvolution"
     */
    std::string name() override;

    /**
     * @brief Runs the symbol evolution pass on the given SDFG.
     *
     * Traverses all structured loops in the SDFG and attempts to derive closed-form
     * expressions for loop-dependent scalar symbols.
     *
     * @param builder The SDFG builder providing access to the graph
     * @param analysis_manager The analysis manager for running required analyses
     * @return true if the pass modified the SDFG, false otherwise
     */
    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
