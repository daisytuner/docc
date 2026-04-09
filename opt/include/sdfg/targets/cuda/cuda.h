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

/**
 * @brief CUDA implementation with automatic memory transfers
 * Used for CUBLAS, memset, and other CUDA-accelerated library nodes
 */
inline data_flow::ImplementationType ImplementationType_CUDAWithTransfers{"CUDAWithTransfers"};

/**
 * @brief CUDA implementation without memory transfers
 * Used for CUBLAS, memset, and other CUDA-accelerated library nodes assuming data is already on GPU
 */
inline data_flow::ImplementationType ImplementationType_CUDAWithoutTransfers{"CUDAWithoutTransfers"};

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

void check_cuda_kernel_launch_errors(
    codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension, bool instrumented
);

} // namespace cuda
} // namespace sdfg
