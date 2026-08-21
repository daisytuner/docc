#pragma once

#include <string>

#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/targets/gpu/gpu_types.h"

namespace sdfg {
namespace rocm {

inline std::string ROCM_DEVICE_PREFIX = "__daisy_hip_";

constexpr int ROCM_WARP_SIZE = 64;

/**
 * @brief ROCM implementation with automatic memory transfers
 * Used for ROCm BLAS, memset, and other ROCm-accelerated library nodes
 */
inline data_flow::ImplementationType ImplementationType_ROCMWithTransfers{"ROCMWithTransfers"};

/**
 * @brief ROCM implementation without memory transfers
 * Used for ROCm BLAS, memset, and other ROCm-accelerated library nodes assuming data is already on GPU
 */
inline data_flow::ImplementationType ImplementationType_ROCMWithoutTransfers{"ROCMWithoutTransfers"};

// Use shared GPU dimension type
using ROCMDimension = gpu::GPUDimension;

/**
 * @brief ROCM schedule type inheriting shared GPU functionality
 * Provides ROCM-specific value() and default block size (64 for wavefront size)
 */
class ScheduleType_ROCM_Offload : public gpu::ScheduleType_GPU_Offload {
public:
    static const std::string value() { return "ROCM_Offload"; }
};

/**
 * @brief ROCM schedule type inheriting shared GPU functionality
 * Provides ROCM-specific value() and default block size (64 for wavefront size)
 * @deprecated This class is deprecated and will be removed in future versions. Use the new GPU schedule type classes
 * instead.
 */
class ScheduleType_ROCM : public gpu::ScheduleType_GPU_Base<ScheduleType_ROCM> {
public:
    static const std::string value() { return "ROCM"; }
    static symbolic::Integer default_block_size_x() { return symbolic::integer(64); }
};

inline codegen::TargetType TargetType_ROCM_Offload{ScheduleType_ROCM_Offload::value()};

inline codegen::TargetType TargetType_ROCM{ScheduleType_ROCM::value()};


void rocm_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
);

bool do_rocm_error_checking();

/// Wavefront width of the target device, queried once at codegen time.
///
/// RDNA parts (gfx10xx/11xx/12xx) execute wave32 while CDNA/GCN (gfx9xx) execute
/// wave64, so the value that drives warp-shuffle reductions and launch geometry
/// cannot be a compile-time constant. Resolved from the `DOCC_ROCM_WAVEFRONT_SIZE`
/// override, else the present device via `rocminfo`, falling back to
/// `ROCM_WARP_SIZE`. The result is cached.
int rocm_wavefront_size();

void check_rocm_kernel_launch_errors(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension);

} // namespace rocm
} // namespace sdfg
