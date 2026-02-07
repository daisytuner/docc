#pragma once

#include <string>

#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_control_flow/map.h"

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

enum CUDADimension { X = 0, Y = 1, Z = 2 };

class ScheduleType_CUDA {
public:
    static void dimension(structured_control_flow::ScheduleType& schedule, const CUDADimension& dimension);
    static CUDADimension dimension(const structured_control_flow::ScheduleType& schedule);
    static void block_size(structured_control_flow::ScheduleType& schedule, const symbolic::Expression block_size);
    static symbolic::Integer block_size(const structured_control_flow::ScheduleType& schedule);
    static bool nested_sync(const structured_control_flow::ScheduleType& schedule);
    static void nested_sync(structured_control_flow::ScheduleType& schedule, const bool nested_sync);
    static const std::string value() { return "CUDA"; }
    static structured_control_flow::ScheduleType create() {
        auto schedule_type =
            structured_control_flow::ScheduleType(value(), structured_control_flow::ScheduleTypeCategory::Offloader);
        dimension(schedule_type, CUDADimension::X);
        return schedule_type;
    }
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
