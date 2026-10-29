#include "sdfg/passes/redundant_zero_init_elimination_pass.h"

#include <vector>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/redundant_zero_init_elimination.h"

namespace sdfg {
namespace passes {

bool RedundantZeroInitEliminationPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Collect first: applying the transformation removes loops and invalidates the loop analysis.
    // Annotated init nests are disjoint, so removing one does not affect the others.
    std::cerr << "Running RedundantZeroInitEliminationPass..." << std::endl;
    std::vector<structured_control_flow::StructuredLoop*> candidates;
    for (auto* node : analysis_manager.get<analysis::LoopAnalysis>().loops()) {
        auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(node);
        if (loop != nullptr && structured_control_flow::ScheduleType_ZeroInit::is_set(loop->schedule_type())) {
            candidates.push_back(loop);
        }
    }

    bool applied = false;
    for (auto* loop : candidates) {
        transformations::RedundantZeroInitElimination transformation(*loop);
        if (transformation.can_be_applied(builder, analysis_manager)) {
            transformation.apply(builder, analysis_manager);
            applied = true;
        }
    }
    return applied;
}

} // namespace passes
} // namespace sdfg
