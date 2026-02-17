#pragma once

#include <string>

#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
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

/**
 * @brief HIP schedule type inheriting shared GPU functionality
 * Provides HIP-specific value() and default block size (64 for wavefront size)
 */
class ScheduleType_HIP : public gpu::ScheduleType_GPU_Base<ScheduleType_HIP> {
public:
    static const std::string value() { return "HIP"; }
    static symbolic::Integer default_block_size_x() { return symbolic::integer(64); }
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
