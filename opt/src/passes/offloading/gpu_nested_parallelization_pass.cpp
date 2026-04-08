#include "sdfg/passes/offloading/gpu_nested_parallelization_pass.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/transformations/offloading/cuda_parallelize_nested_map.h"
#include "sdfg/transformations/offloading/rocm_parallelize_nested_map.h"

namespace sdfg {
namespace passes {

GPUNestedParallelizationPass::GPUNestedParallelizationPass(
    const std::vector<structured_control_flow::Map*>& maps, GPUTarget target, size_t block_size
)
    : maps_(maps), target_(target), block_size_(block_size) {}

bool GPUNestedParallelizationPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (maps_.empty()) {
        return false;
    }

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Phase 1: Collect all applicable nested maps
    std::vector<structured_control_flow::Map*> candidates;

    for (auto* map : maps_) {
        auto descendants = loop_analysis.descendants(map);
        for (auto* descendant : descendants) {
            if (auto* nested_map = dynamic_cast<structured_control_flow::Map*>(descendant)) {
                bool applicable = false;
                if (target_ == GPUTarget::CUDA) {
                    transformations::CUDAParallelizeNestedMap transform(*nested_map, block_size_);
                    applicable = transform.can_be_applied(builder, analysis_manager);
                } else {
                    transformations::ROCMParallelizeNestedMap transform(*nested_map, block_size_);
                    applicable = transform.can_be_applied(builder, analysis_manager);
                }
                if (applicable) {
                    candidates.push_back(nested_map);
                }
            }
        }
    }

    if (candidates.empty()) {
        return false;
    }

    // Phase 2: Apply all parallelizations
    for (auto* nested_map : candidates) {
        if (target_ == GPUTarget::CUDA) {
            transformations::CUDAParallelizeNestedMap transform(*nested_map, block_size_);
            transform.apply(builder, analysis_manager);
        } else {
            transformations::ROCMParallelizeNestedMap transform(*nested_map, block_size_);
            transform.apply(builder, analysis_manager);
        }
    }

    return true;
}

} // namespace passes
} // namespace sdfg
