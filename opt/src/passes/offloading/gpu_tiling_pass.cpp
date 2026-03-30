#include "sdfg/passes/offloading/gpu_tiling_pass.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/offloading/gpu_tiling.h"

namespace sdfg {
namespace passes {

GPUTilingPass::GPUTilingPass(const std::vector<structured_control_flow::Map*>& maps, size_t tile_size)
    : maps_(maps), tile_size_(tile_size) {}

bool GPUTilingPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (maps_.empty()) {
        return false;
    }

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Phase 1: Collect all applicable tiling targets
    std::vector<structured_control_flow::StructuredLoop*> candidates;

    for (auto* map : maps_) {
        for (auto* descendant : loop_analysis.descendants(map)) {
            if (auto* target_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(descendant)) {
                transformations::GPUTiling tiling(*target_loop, tile_size_);
                if (tiling.can_be_applied(builder, analysis_manager)) {
                    candidates.push_back(target_loop);
                }
            }
        }
    }

    if (candidates.empty()) {
        return false;
    }

    // Phase 2: Apply all tilings
    for (auto* target_loop : candidates) {
        transformations::GPUTiling tiling(*target_loop, tile_size_);
        tiling.apply(builder, analysis_manager);
    }

    return true;
}

} // namespace passes
} // namespace sdfg
