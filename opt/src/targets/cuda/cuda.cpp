#include "sdfg/targets/cuda/cuda.h"

#include <cstdlib>
#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/assumptions_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/exceptions.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/type.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>
#include <string>

namespace sdfg {
namespace cuda {

void ScheduleType_CUDA::dimension(structured_control_flow::ScheduleType& schedule, const CUDADimension& dimension) {
    schedule.set_property("dimension", std::to_string(dimension));
}
CUDADimension ScheduleType_CUDA::dimension(const structured_control_flow::ScheduleType& schedule) {
    return static_cast<CUDADimension>(std::stoi(schedule.properties().at("dimension")));
}
void ScheduleType_CUDA::block_size(structured_control_flow::ScheduleType& schedule, const symbolic::Expression block_size) {
    serializer::JSONSerializer serializer;
    schedule.set_property("block_size", serializer.expression(block_size));
}

symbolic::Integer ScheduleType_CUDA::block_size(const structured_control_flow::ScheduleType& schedule) {
    if (schedule.properties().find("block_size") == schedule.properties().end()) {
        if (dimension(schedule) == CUDADimension::X) {
            return symbolic::integer(32);
        } else if (dimension(schedule) == CUDADimension::Y) {
            return symbolic::integer(8);
        } else if (dimension(schedule) == CUDADimension::Z) {
            return symbolic::integer(4);
        } else {
            throw InvalidSDFGException("Invalid CUDA dimension");
        }
    }
    std::string expr_str = schedule.properties().at("block_size");
    return symbolic::integer(std::stoi(expr_str));
}

bool ScheduleType_CUDA::nested_sync(const structured_control_flow::ScheduleType& schedule) {
    if (schedule.properties().find("nested_sync") == schedule.properties().end()) {
        return false;
    }
    std::string val = schedule.properties().at("nested_sync");
    return val == "true";
}

void ScheduleType_CUDA::nested_sync(structured_control_flow::ScheduleType& schedule, const bool nested_sync) {
    schedule.set_property("nested_sync", nested_sync ? "true" : "false");
}

void cuda_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
) {
    if (!do_cuda_error_checking()) {
        return;
    }
    stream << "if (" << status_variable << " != cudaSuccess) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << language_extension.external_prefix()
           << "fprintf(stderr, \"CUDA error: %s File: %s, Line: %d\\n\", cudaGetErrorString(" << status_variable
           << "), __FILE__, __LINE__);" << std::endl;
    stream << language_extension.external_prefix() << "exit(EXIT_FAILURE);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

bool do_cuda_error_checking() {
    auto env = getenv("DOCC_CUDA_DEBUG");
    if (env == nullptr) {
        return false;
    }
    std::string env_str(env);
    std::transform(env_str.begin(), env_str.end(), env_str.begin(), ::tolower);
    if (env_str == "1" || env_str == "true") {
        return true;
    }
    return false;
}

void check_cuda_kernel_launch_errors(
    codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension, bool instrumented
) {
    if (!do_cuda_error_checking() && !instrumented) {
        return;
    }
    stream << "cudaError_t launch_err = cudaDeviceSynchronize();" << std::endl;
    cuda_error_checking(stream, language_extension, "launch_err");
    stream << "launch_err = cudaGetLastError();" << std::endl;
    cuda_error_checking(stream, language_extension, "launch_err");
}

} // namespace cuda
} // namespace sdfg
