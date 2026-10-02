#pragma once

#include <ostream>
#include <stdint.h>
#include <string>

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fragment.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

class GpuArch {
public:
    static constexpr const char* ARCH_PROPERTY = "ARCH";

    GpuArch() {
    }
    virtual ~GpuArch() = default;

    virtual std::string unique_id() const = 0;
    virtual std::string name() const = 0;
    virtual int per_cu_threads() const = 0;

    virtual const GpuMmaSupport* mma_support() const = 0;

    virtual structured_control_flow::ScheduleType create_schedule_type() const = 0;

    static const GpuArch* get_from_schedule_type(const structured_control_flow::ScheduleType& schedule);
};

namespace util {

/// Runs a command and captures its standard output. Returns std::nullopt if the
/// command could not be launched.
std::optional<std::string> run_and_capture(const std::string& command);

/// Trims surrounding whitespace from a string.
std::string trim(const std::string& raw);

} // namespace util

} // namespace sdfg::gpu
