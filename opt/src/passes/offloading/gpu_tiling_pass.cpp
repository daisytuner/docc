#include "sdfg/passes/offloading/gpu_tiling_pass.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/offloading/gpu_tiling.h"
#include "sdfg/transformations/offloading/kernel_local_storage.h"

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
    auto& users = analysis_manager.get<analysis::Users>();

    // Find targets for normal loop tiling
    for (auto* map : maps_) {
        for (auto* descendant : loop_analysis.descendants(map)) {
            auto* struc_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(descendant);
            if (!struc_loop) continue;


            // Pre-filter: check if any container used in the loop is a KLS candidate
            analysis::UsersView users_view(users, struc_loop->root());
            bool has_kls_candidate = false;
            for (auto& use : users_view.reads()) {
                auto element = use->element();
                if (!dynamic_cast<data_flow::AccessNode*>(element)) {
                    continue;
                }
                auto access_node = dynamic_cast<data_flow::AccessNode*>(element);
                if (transformations::KernelLocalStorage::
                        is_candidate(*struc_loop, access_node->data(), builder, analysis_manager)) {
                    has_kls_candidate = true;
                    break;
                }
            }
            if (!has_kls_candidate) continue;

            transformations::LoopTiling tiling(*struc_loop, tile_size_);
            if (tiling.can_be_applied(builder, analysis_manager)) {
                candidates.push_back(struc_loop);
            }
        }
    }

    std::vector<structured_control_flow::StructuredLoop*> tilable_loops;

    for (auto* map : candidates) {
        for (auto* descendant : loop_analysis.descendants(map)) {
            if (auto* target_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(descendant)) {
                transformations::GPUTiling tiling(*target_loop, tile_size_);
                if (tiling.can_be_applied(builder, analysis_manager)) {
                    tilable_loops.push_back(target_loop);
                }
            }
        }
    }

    if (tilable_loops.empty()) {
        return false;
    }

    // Phase 2: Apply all tilings
    for (auto* target_loop : tilable_loops) {
        transformations::GPUTiling tiling(*target_loop, tile_size_);
        tiling.apply(builder, analysis_manager);
    }

    return true;
}

} // namespace passes
} // namespace sdfg
