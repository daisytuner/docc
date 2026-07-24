#pragma once

#include <string>

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Ensures every structured loop has a unique induction variable.
 *
 * Traverses all structured loops in the SDFG (via the structured visitor). Each
 * loop's induction variable is recorded in a set of already-used names. If a loop
 * reuses an induction variable that has already been seen (e.g. two sibling or
 * nested loops both named `i`), and that induction variable is genuinely read
 * inside the loop, the loop's induction variable is renamed to a fresh container
 * and every use of the old symbol within that loop (init, update, condition and
 * body) is replaced.
 *
 * Whether the induction variable is genuinely read is decided with the Users
 * analysis: there must be a read user that depends on the loop's update user and
 * that is not the loop-control use of another loop (i.e. an actual data read).
 *
 * Disambiguating shared induction variables is required for downstream analyses
 * (e.g. argument-size/subset inference for offloading) that key on the symbol of
 * a loop variable and would otherwise treat a nested loop variable as a free symbol.
 */
class UniqueLoopIndvars : public Pass {
public:
    UniqueLoopIndvars();

    std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
