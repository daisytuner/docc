#pragma once

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/transformations/offloading/gpu_offload_transform.h"

namespace sdfg {
namespace cuda {


class CUDAOffloadTransform : public gpu::GPUOffloadTransform<CUDADataOffloadingNode> {
public:
    explicit CUDAOffloadTransform(
        structured_control_flow::StructuredLoop& loop,
        symbolic::Integer parallel_size,
        gpu::TargetLevel target_level = gpu::TargetLevel::X_GRID,
        bool allow_dynamic_sizes = false
    )
        : gpu::GPUOffloadTransform<CUDADataOffloadingNode>(loop, parallel_size, target_level, allow_dynamic_sizes) {};

    std::string name() const override;

protected:
    types::StorageType local_device_storage_type() override;

    types::StorageType global_device_storage_type(symbolic::Expression arg_size) override;

    ScheduleType transformed_schedule_type() override;

    std::string copy_prefix() override;
};

} // namespace cuda
} // namespace sdfg
