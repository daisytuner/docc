#pragma once

#include <string>

#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace printf_target {

/// Prefix for device copies to differentiate from host arguments
inline std::string PRINTF_DEVICE_PREFIX = "__printf_device_";

/**
 * @brief Schedule type for the printf debug target
 *
 * This schedule type is used for debugging/tracing purposes.
 * Instead of generating actual kernel code, it generates printf
 * statements that trace the execution flow.
 */
class ScheduleType_Printf {
public:
    /// Get the target stream for this schedule
    static std::string target_stream(const structured_control_flow::ScheduleType& schedule);

    /// Set the target stream for this schedule
    static void target_stream(structured_control_flow::ScheduleType& schedule, const std::string& stream);

    /// Returns the unique identifier for this schedule type
    static const std::string value() { return "Printf"; }

    /// Creates a default Printf schedule
    static structured_control_flow::ScheduleType create() {
        auto schedule_type = structured_control_flow::ScheduleType(value(), structured_control_flow::ScheduleTypeCategory::Offloader);
        return schedule_type;
    }
};

/// Target type constant for the printf target
inline codegen::TargetType TargetType_Printf{ScheduleType_Printf::value()};

} // namespace printf_target
} // namespace sdfg
