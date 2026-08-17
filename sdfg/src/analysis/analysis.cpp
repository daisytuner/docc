#include "sdfg/analysis/analysis.h"

namespace sdfg {
namespace analysis {

Analysis::Analysis(StructuredSDFG& sdfg)
    : sdfg_(sdfg) {

      };

AnalysisManager::AnalysisManager(StructuredSDFG& sdfg)
    : sdfg_(sdfg) {

      };

AnalysisManager::AnalysisManager(StructuredSDFG& sdfg, const symbolic::Assumptions& additional_assumptions)
    : sdfg_(sdfg), additional_assumptions_(additional_assumptions) {

      };

void AnalysisManager::invalidate_all() {
    if (cache_.empty()) {
        return;
    }
    passes::CompileStatistics* stats = nullptr;
    if (passes::CompileStatistics::enabled()) {
        stats = &passes::CompileStatistics::instance();
        stats->enter_scope("invalidate_all", STATS_SCOPE);
        stats->add_metric("invalidations", cache_.size());
    }

    cache_.clear();

    if (stats) {
        stats->exit_scope();
    }
}

} // namespace analysis
} // namespace sdfg
