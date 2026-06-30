#pragma once

#include <vector>

#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/reduce.h"

namespace sdfg {
namespace passes {

/**
 * @brief Classifies parallelizable `For` loops and lowers them to `Map` or
 * `Reduce`.
 *
 * A `For` loop is examined against a single shared set of legality criteria
 * (monotonic bound, no side effects, no early `Return`, induction variable dead
 * after the loop, false dependencies confined to loop-local storage). The only
 * thing that distinguishes the two outcomes is how the loop's loop-carried
 * read-write dependencies are resolved:
 *   - none            -> the iterations are independent  -> `Map`
 *   - all reductions  -> the only carried dependencies are reorderable
 *                        accumulations (recognized by the loop-carried
 *                        dependency analysis) -> `Reduce`
 *   - otherwise       -> a genuine hazard remains; the loop is left as a `For`.
 */
class ForClassificationPass : public Pass {
public:
    enum class Classification { None, Map, Reduce };

private:
    Classification classify(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::For& for_stmt,
        std::vector<structured_control_flow::ReductionInfo>& reductions
    );

public:
    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
