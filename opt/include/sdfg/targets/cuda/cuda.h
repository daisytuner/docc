#pragma once

#include <string>

#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/targets/gpu/gpu_types.h"

namespace sdfg {
namespace cuda {

inline std::string CUDA_DEVICE_PREFIX = "__daisy_cuda_";

namespace blas {
/**
 * @brief CUBLAS implementation with automatic memory transfers
 * Uses NVIDIA CUBLAS with automatic host-device data transfers
 */
inline data_flow::ImplementationType ImplementationType_CUBLASWithTransfers{"CUBLASWithTransfers"};

/**
 * @brief CUBLAS implementation without memory transfers
 * Uses NVIDIA CUBLAS assuming data is already on GPU
 */
inline data_flow::ImplementationType ImplementationType_CUBLASWithoutTransfers{"CUBLASWithoutTransfers"};
} // namespace blas

// Use shared GPU dimension type
using CUDADimension = gpu::GPUDimension;

/**
 * @brief CUDA schedule type inheriting shared GPU functionality
 * Provides CUDA-specific value() and default block size (32 for warp size)
 */
class ScheduleType_CUDA : public gpu::ScheduleType_GPU_Base<ScheduleType_CUDA> {
public:
    static const std::string value() { return "CUDA"; }
    static symbolic::Integer default_block_size_x() { return symbolic::integer(32); }
};

inline codegen::TargetType TargetType_CUDA{ScheduleType_CUDA::value()};


void cuda_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
);

bool do_cuda_error_checking();

void check_cuda_kernel_launch_errors(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension);

} // namespace cuda
} // namespace sdfg
