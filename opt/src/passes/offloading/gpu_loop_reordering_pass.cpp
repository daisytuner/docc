#include "sdfg/passes/offloading/gpu_loop_reordering_pass.h"

#include "sdfg/transformations/offloading/gpu_loop_reordering.h"

namespace sdfg {
namespace passes {

GPULoopReorderingPass::GPULoopReorderingPass(const std::vector<structured_control_flow::Map*>& maps) : maps_(maps) {}

bool GPULoopReorderingPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (maps_.empty()) {
        return false;
    }

    // Phase 1: Collect applicable reorderings
    struct ReorderingCandidate {
        structured_control_flow::Map* map;
    };
    std::vector<ReorderingCandidate> candidates;

    for (auto* map : maps_) {
        transformations::GPULoopReordering reordering(*map);
        if (reordering.can_be_applied(builder, analysis_manager)) {
            candidates.push_back({map});
        }
    }

    if (candidates.empty()) {
        return false;
    }

    // Phase 2: Apply all reorderings
    for (auto& candidate : candidates) {
        transformations::GPULoopReordering reordering(*candidate.map);
        reordering.apply(builder, analysis_manager);
    }

    return true;
}

} // namespace passes
} // namespace sdfg
