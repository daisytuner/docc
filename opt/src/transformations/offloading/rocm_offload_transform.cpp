#include "sdfg/transformations/offloading/rocm_offload_transform.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/transformations/transformation.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace rocm {

std::string ROCMOffloadTransform::name() const {
    return "ROCMOffloadTransform";
}

ROCMOffloadTransform ROCMOffloadTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
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

    return ROCMOffloadTransform(*loop, parallel_size, target_level);
}

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

std::string ROCMOffloadTransform::copy_prefix() {
    return ROCM_DEVICE_PREFIX;
}

} // namespace rocm
} // namespace sdfg
