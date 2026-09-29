#include "sdfg/passes/tiling_pass.h"

#include "sdfg/transformations/loop_tiling.h"

namespace sdfg {
namespace passes {

TilingPass::TilingPass(std::vector<structured_control_flow::StructuredLoop*>& loops, size_t tile_size)
    : loops_(loops), tile_size_(tile_size) {
}

bool TilingPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (loops_.empty() || tile_size_ <= 1) {
        return false;
    }

    // Phase 1: Collect applicable tilings
    struct TilingCandidate {
        size_t loop_index;
        transformations::LoopTiling tiling;
    };
    std::vector<TilingCandidate> candidates;

    for (size_t i = 0; i < loops_.size(); ++i) {
        transformations::LoopTiling tiling(*loops_[i], tile_size_);
        if (tiling.can_be_applied(builder, analysis_manager)) {
            candidates.push_back({i, std::move(tiling)});
        }
    }

    if (candidates.empty()) {
        return false;
    }

    // Phase 2: Apply all tilings
    for (auto& candidate : candidates) {
        candidate.tiling.apply(builder, analysis_manager);
        // Update the loop pointer to the new outer loop
        loops_[candidate.loop_index] = candidate.tiling.outer_loop();
    }
    analysis_manager.invalidate_all();

    return true;
}

} // namespace passes
} // namespace sdfg
