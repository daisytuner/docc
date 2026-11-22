#include "sdfg/loops/passes/map_delinearization.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/loops/transformations/map_delinearize.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace loops {

bool MapDelinearizationPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool applied = false;
    // Delinearizing a map moves the loops inside it, which stay valid.
    for (auto* loop : analysis_manager.get<analysis::LoopAnalysis>().loops()) {
        auto map = dynamic_cast<structured_control_flow::Map*>(loop);
        if (!map) {
            continue;
        }
        loops::transformations::MapDelinearize delinearize(*map);
        if (delinearize.can_be_applied(builder, analysis_manager)) {
            delinearize.apply(builder, analysis_manager);
            applied = true;
        }
    }
    return applied;
}

} // namespace loops
} // namespace sdfg
