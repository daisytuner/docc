#include "sdfg/targets/hip/hip.h"

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
namespace hip {

void ScheduleType_HIP::dimension(structured_control_flow::ScheduleType& schedule, const gpu::GPUDimension& dimension) {
    schedule.set_property("dimension", std::to_string(dimension));
}
gpu::GPUDimension ScheduleType_HIP::dimension(const structured_control_flow::ScheduleType& schedule) {
    return static_cast<gpu::GPUDimension>(std::stoi(schedule.properties().at("dimension")));
}
void ScheduleType_HIP::block_size(structured_control_flow::ScheduleType& schedule, const symbolic::Expression block_size) {
    serializer::JSONSerializer serializer;
    schedule.set_property("block_size", serializer.expression(block_size));
}

symbolic::Integer ScheduleType_HIP::block_size(const structured_control_flow::ScheduleType& schedule) {
    if (schedule.properties().find("block_size") == schedule.properties().end()) {
        if (dimension(schedule) == gpu::GPUDimension::X) {
            return symbolic::integer(64); // HIP default wavefront size is 64
        } else if (dimension(schedule) == gpu::GPUDimension::Y) {
            return symbolic::integer(8);
        } else if (dimension(schedule) == gpu::GPUDimension::Z) {
            return symbolic::integer(4);
        } else {
            throw InvalidSDFGException("Invalid HIP dimension");
        }
    }
    std::string expr_str = schedule.properties().at("block_size");
    return symbolic::integer(std::stoi(expr_str));
}

bool ScheduleType_HIP::nested_sync(const structured_control_flow::ScheduleType& schedule) {
    if (schedule.properties().find("nested_sync") == schedule.properties().end()) {
        return false;
    }
    std::string val = schedule.properties().at("nested_sync");
    return val == "true";
}

void ScheduleType_HIP::nested_sync(structured_control_flow::ScheduleType& schedule, const bool nested_sync) {
    schedule.set_property("nested_sync", nested_sync ? "true" : "false");
}

void hip_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
) {
    if (!do_hip_error_checking()) {
        return;
    }
    stream << "if (" << status_variable << " != hipSuccess) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << language_extension.external_prefix()
           << "fprintf(stderr, \"HIP error: %s File: %s, Line: %d\\n\", hipGetErrorString(" << status_variable
           << "), __FILE__, __LINE__);" << std::endl;
    stream << language_extension.external_prefix() << "exit(EXIT_FAILURE);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

bool do_hip_error_checking() {
    auto env = getenv("DOCC_HIP_DEBUG");
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

void check_hip_kernel_launch_errors(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    if (!do_hip_error_checking()) {
        return;
    }
    stream << "hipError_t launch_err = hipDeviceSynchronize();" << std::endl;
    hip_error_checking(stream, language_extension, "launch_err");
    stream << "launch_err = hipGetLastError();" << std::endl;
    hip_error_checking(stream, language_extension, "launch_err");
}

} // namespace hip
} // namespace sdfg
