#pragma once

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief [DEPRECATED] Monolithic GPU tiling transformation.
 *
 * Prefer the composable pipeline instead:
 *   1. transformations::LoopTiling             — strip-mine the target loop
 *   2. transformations::CUDAParallelizeNestedMap / cuda::CUDATransform
 *                                              — assign GPU schedules
 *   3. transformations::InLocalStorage  (NV_Shared) — stage read tiles
 *      transformations::OutLocalStorage (NV_Shared) — stage write tiles
 *   4. passes::SyncConditionPropagation         — guard out-of-bounds threads
 *
 * See `docc/opt/tests/optimizations/gpu_kernels_test.cpp` for a worked
 * example (GEMM). KernelLocalStorage and GPUTilingPass are deprecated for
 * the same reason.
 *
 * The legacy transformation is retained for autotuning search spaces and
 * existing schedulers that have not yet been migrated.
 */
class [[deprecated(
    "Use LoopTiling + CUDA/ROCm parallelize + In/OutLocalStorage + SyncConditionPropagation. See gpu_kernels_test.cpp."
)]] GPUTiling : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    size_t size_;
    bool applied_ = false;

    structured_control_flow::StructuredLoop* inner_loop_ = nullptr;
    structured_control_flow::StructuredLoop* outer_loop_ = nullptr;
    std::set<std::string> target_containers_;

public:
    GPUTiling(structured_control_flow::StructuredLoop& loop, size_t size);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static GPUTiling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    structured_control_flow::StructuredLoop* inner_loop();
    structured_control_flow::StructuredLoop* outer_loop();
};

} // namespace transformations
} // namespace sdfg
