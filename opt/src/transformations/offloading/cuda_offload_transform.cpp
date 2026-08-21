#include "sdfg/transformations/offloading/cuda_offload_transform.h"

#include "sdfg/targets/cuda/cuda.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace cuda {

std::string CUDAOffloadTransform::name() const { return "CUDAOffloadTransform"; }

types::StorageType CUDAOffloadTransform::local_device_storage_type() {
    return types::StorageType(
        "NV_Generic",
        SymEngine::null,
        types::StorageType::AllocationType::Unmanaged,
        types::StorageType::AllocationType::Unmanaged
    );
}

types::StorageType CUDAOffloadTransform::global_device_storage_type(symbolic::Expression arg_size) {
    return types::StorageType(
        "NV_Generic",
        arg_size,
        types::StorageType::AllocationType::Unmanaged,
        types::StorageType::AllocationType::Unmanaged
    );
}

ScheduleType CUDAOffloadTransform::transformed_schedule_type() {
    return ScheduleType_CUDA::create<ScheduleType_CUDA_Offload>(target_level_, parallel_size_);
}

std::string CUDAOffloadTransform::copy_prefix() { return CUDA_DEVICE_PREFIX; }

} // namespace cuda
} // namespace sdfg
