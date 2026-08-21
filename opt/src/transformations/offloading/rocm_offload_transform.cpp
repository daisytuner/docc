#include "sdfg/transformations/offloading/rocm_offload_transform.h"

#include "sdfg/targets/rocm/rocm.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace rocm {

std::string ROCMOffloadTransform::name() const { return "ROCMOffloadTransform"; }

types::StorageType ROCMOffloadTransform::local_device_storage_type() {
    return types::StorageType(
        "AMD_Generic",
        SymEngine::null,
        types::StorageType::AllocationType::Unmanaged,
        types::StorageType::AllocationType::Unmanaged
    );
}

types::StorageType ROCMOffloadTransform::global_device_storage_type(symbolic::Expression arg_size) {
    return types::StorageType(
        "AMD_Generic",
        arg_size,
        types::StorageType::AllocationType::Unmanaged,
        types::StorageType::AllocationType::Unmanaged
    );
}

ScheduleType ROCMOffloadTransform::transformed_schedule_type() {
    return ScheduleType_ROCM_Offload::create<ScheduleType_ROCM_Offload>(target_level_, parallel_size_);
}

std::string ROCMOffloadTransform::copy_prefix() { return ROCM_DEVICE_PREFIX; }

} // namespace rocm
} // namespace sdfg
