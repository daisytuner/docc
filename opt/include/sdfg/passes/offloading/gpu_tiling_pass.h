#pragma once

#include <vector>
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {

/**
 * @brief [DEPRECATED] Phased GPU tiling pass.
 *
 * Drives the legacy `transformations::GPUTiling` +
 * `transformations::KernelLocalStorage` pair. Both are deprecated. New code
 * should compose `LoopTiling`, `CUDAParallelizeNestedMap` / `cuda::CUDATransform`,
 * `InLocalStorage` / `OutLocalStorage` (with `NV_Shared`), and
 * `passes::SyncConditionPropagation` directly — see
 * `docc/opt/tests/optimizations/gpu_kernels_test.cpp` for a worked example.
 *
 * Retained for the existing CUDA and ROCm schedulers that have not yet been
 * migrated.
 *
 * Given a set of outer maps, finds all descendant structured loops and applies
 * GPU tiling in two phases:
 * 1. can_be_applied phase: collects all loops where tiling is applicable
 * 2. apply phase: applies tiling to all collected loops
 */
class [[deprecated(
    "Use LoopTiling + CUDA/ROCm parallelize + In/OutLocalStorage + SyncConditionPropagation. See gpu_kernels_test.cpp."
)]] GPUTilingPass : public Pass {
private:
    const std::vector<structured_control_flow::Map*>& maps_;
    size_t tile_size_;

public:
    GPUTilingPass(const std::vector<structured_control_flow::Map*>& maps, size_t tile_size);
    ~GPUTilingPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override { return "GPUTilingPass"; }
};

} // namespace passes
} // namespace sdfg
