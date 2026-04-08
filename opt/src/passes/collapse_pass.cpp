#include "sdfg/passes/collapse_pass.h"

#include "sdfg/transformations/collapse_to_depth.h"

namespace sdfg {
namespace passes {

CollapsePass::CollapsePass(std::vector<structured_control_flow::Map*>& maps, size_t target_depth)
    : maps_(maps), target_depth_(target_depth) {}

bool CollapsePass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (maps_.empty() || target_depth_ == 0) {
        return false;
    }

    // Phase 1: Collect applicable collapses
    struct CollapseCandidate {
        size_t map_index;
        transformations::CollapseToDepth collapse;
    };
    std::vector<CollapseCandidate> candidates;

    for (size_t i = 0; i < maps_.size(); ++i) {
        transformations::CollapseToDepth collapse(*maps_[i], target_depth_);
        if (collapse.can_be_applied(builder, analysis_manager)) {
            candidates.push_back({i, std::move(collapse)});
        }
    }

    if (candidates.empty()) {
        return false;
    }

    // Phase 2: Apply all collapses
    for (auto& candidate : candidates) {
        candidate.collapse.apply(builder, analysis_manager);
        // Update the map pointer to the collapsed outer loop
        maps_[candidate.map_index] = candidate.collapse.outer_loop();
    }
    analysis_manager.invalidate_all();

    return true;
}

} // namespace passes
} // namespace sdfg
