#include "sdfg/analysis/analysis.h"

namespace sdfg {
namespace analysis {

Analysis::Analysis(StructuredSDFG& sdfg)
    : sdfg_(sdfg) {

      };

Analysis::Analysis(StructuredSDFG& sdfg, const Options& options) : sdfg_(sdfg), options_(&options) {};

AnalysisManager::AnalysisManager(StructuredSDFG& sdfg)
    : sdfg_(sdfg) {

      };

AnalysisManager::AnalysisManager(StructuredSDFG& sdfg, const symbolic::Assumptions& additional_assumptions)
    : sdfg_(sdfg), additional_assumptions_(additional_assumptions) {

      };

AnalysisManager::AnalysisManager(StructuredSDFG& sdfg, const Options& options) : sdfg_(sdfg), options_(&options) {};

AnalysisManager::
    AnalysisManager(StructuredSDFG& sdfg, const symbolic::Assumptions& additional_assumptions, const Options& options)
    : sdfg_(sdfg), additional_assumptions_(additional_assumptions), options_(&options) {};

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

void AnalysisManager::invalidate_all(structured_control_flow::ControlFlowNode& changed) {
    for (auto it = cache_.begin(); it != cache_.end();) {
        if (it->second->invalidate(changed)) {
            ++it;
        } else {
            it = cache_.erase(it);
        }
    }
}

} // namespace analysis
} // namespace sdfg
