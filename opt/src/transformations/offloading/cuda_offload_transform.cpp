#include "sdfg/transformations/offloading/cuda_offload_transform.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/transformations/transformation.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace cuda {

std::string CUDAOffloadTransform::name() const {
    return "CUDAOffloadTransform";
}

CUDAOffloadTransform CUDAOffloadTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto target_level = gpu::target_level_from_string(desc["parameters"]["target_level"].get<std::string>());
    symbolic::Integer parallel_size =
        SymEngine::rcp_static_cast<const SymEngine::Integer>(symbolic::parse(desc["parameters"]["parallel_size"]));
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(element);
    if (!loop) {
        throw transformations::InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop."
        );
    }

    return CUDAOffloadTransform(*loop, parallel_size, target_level);
}

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
    return ScheduleType_CUDA_Offload::create<ScheduleType_CUDA_Offload>(target_level_, parallel_size_);
}

std::string CUDAOffloadTransform::copy_prefix() {
    return CUDA_DEVICE_PREFIX;
}

} // namespace cuda
} // namespace sdfg
