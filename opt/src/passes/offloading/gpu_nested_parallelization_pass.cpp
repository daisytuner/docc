#include "sdfg/passes/offloading/gpu_nested_parallelization_pass.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/offloading/cuda_parallelize_nested_map.h"
#include "sdfg/transformations/offloading/rocm_parallelize_nested_map.h"

namespace sdfg {
namespace passes {

GPUNestedParallelizationPass::GPUNestedParallelizationPass(
    const std::vector<structured_control_flow::StructuredLoop*>& loops, GPUTarget target, size_t block_size
)
    : loops_(loops), target_(target), block_size_(block_size) {}

bool GPUNestedParallelizationPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (loops_.empty()) {
        return false;
    }

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // Phase 1: Collect all applicable nested loops (loops or reduces)
    std::vector<structured_control_flow::StructuredLoop*> candidates;

    for (auto* loop : loops_) {
        auto descendants = loop_analysis.descendants(loop);
        for (auto* descendant : descendants) {
            if (auto* nested_loop = dyn_cast<structured_control_flow::StructuredLoop*>(descendant)) {
                bool applicable = false;
                if (target_ == GPUTarget::CUDA) {
                    transformations::CUDAParallelizeNestedMap transform(*nested_loop, block_size_);
                    applicable = transform.can_be_applied(builder, analysis_manager);
                } else {
                    transformations::ROCMParallelizeNestedMap transform(*nested_loop, block_size_);
                    applicable = transform.can_be_applied(builder, analysis_manager);
                }
                if (applicable) {
                    candidates.push_back(nested_loop);
                }
            }
        }
    }

    if (candidates.empty()) {
        return false;
    }

    // Phase 2: Apply all parallelizations
    for (auto* nested_loop : candidates) {
        if (target_ == GPUTarget::CUDA) {
            transformations::CUDAParallelizeNestedMap transform(*nested_loop, block_size_);
            transform.apply(builder, analysis_manager);
        } else {
            transformations::ROCMParallelizeNestedMap transform(*nested_loop, block_size_);
            transform.apply(builder, analysis_manager);
        }
    }

    return true;
}

} // namespace passes
} // namespace sdfg
