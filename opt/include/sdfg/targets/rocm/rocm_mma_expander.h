#pragma once

#include "sdfg/targets/gpu/gpu_mma_expander.h"
#include "sdfg/targets/rocm/rocm_arch.h"

namespace sdfg::gpu::rocm {

class RocmMmaExpander : public GpuMmaExpander {
    const RocmArch& arch_;

protected:
    GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const override;
    bool matches_possible_mma_pattern(const math::tensor::MatMulNode& node) const override;
    ScheduleType get_schedule_type(gpu::TargetLevel dim, const symbolic::Integer& size) const override;
    void set_implementation_type_mma(math::tensor::MatMulNode& node, const GpuMmaTiling& mma_tiling) const override;

public:
    RocmMmaExpander(const RocmArch& arch) : GpuMmaExpander(), arch_(arch) {}
};

} // namespace sdfg::gpu::rocm
