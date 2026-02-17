#pragma once

#include <string>

#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/targets/gpu/gpu_types.h"

namespace sdfg {
namespace hip {

inline std::string HIP_DEVICE_PREFIX = "__daisy_hip_";

namespace blas {
/**
 * @brief HIPBLAS implementation with automatic memory transfers
 * Uses AMD HIPBLAS with automatic host-device data transfers
 */
inline data_flow::ImplementationType ImplementationType_HIPBLASWithTransfers{"HIPBLASWithTransfers"};

/**
 * @brief HIPBLAS implementation without memory transfers
 * Uses AMD HIPBLAS assuming data is already on GPU
 */
inline data_flow::ImplementationType ImplementationType_HIPBLASWithoutTransfers{"HIPBLASWithoutTransfers"};
} // namespace blas

// Use shared GPU dimension type
using HIPDimension = gpu::GPUDimension;

class ScheduleType_HIP {
public:
    static void dimension(structured_control_flow::ScheduleType& schedule, const gpu::GPUDimension& dimension);
    static gpu::GPUDimension dimension(const structured_control_flow::ScheduleType& schedule);
    static void block_size(structured_control_flow::ScheduleType& schedule, const symbolic::Expression block_size);
    static symbolic::Integer block_size(const structured_control_flow::ScheduleType& schedule);
    static bool nested_sync(const structured_control_flow::ScheduleType& schedule);
    static void nested_sync(structured_control_flow::ScheduleType& schedule, const bool nested_sync);
    static const std::string value() { return "HIP"; }
    static structured_control_flow::ScheduleType create() {
        auto schedule_type =
            structured_control_flow::ScheduleType(value(), structured_control_flow::ScheduleTypeCategory::Offloader);
        dimension(schedule_type, gpu::GPUDimension::X);
        return schedule_type;
    }
};

inline codegen::TargetType TargetType_HIP{ScheduleType_HIP::value()};


void hip_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
);

bool do_hip_error_checking();

void check_hip_kernel_launch_errors(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension);

} // namespace hip
} // namespace sdfg
