#pragma once

#include <sdfg/structured_control_flow/structured_loop.h>

#include <stdexcept>

namespace sdfg {
namespace axelera {

/// Lanes of a PWORD, the vector unit every Axelera stream moves per beat.
inline constexpr size_t PWORD_LANES = 64;

/**
 * Hardware level an AXELERA loop maps to, outermost first: a Core loop distributes contiguous chunks over AI
 * cores, a Stream loop streams PWORD blocks through one datapath, and a Lane loop covers the lanes of one PWORD.
 */
enum class AxeleraLevel { Core, Stream, Lane };

inline std::string to_string(AxeleraLevel level) {
    switch (level) {
        case AxeleraLevel::Core:
            return "core";
        case AxeleraLevel::Stream:
            return "stream";
        case AxeleraLevel::Lane:
            return "lane";
    }
    throw std::invalid_argument("Invalid AxeleraLevel");
}

class ScheduleType_Axelera {
public:
    static const std::string value() {
        return "AXELERA";
    }

    /// @param cores AI cores a Core loop is split across; ignored for the other levels.
    static structured_control_flow::ScheduleType create(AxeleraLevel level, int64_t cores = 1) {
        auto category = level == AxeleraLevel::Lane ? structured_control_flow::ScheduleTypeCategory::Vectorizer
                                                    : structured_control_flow::ScheduleTypeCategory::Offloader;
        structured_control_flow::ScheduleType schedule(value(), category);
        schedule.set_property("level", to_string(level));
        if (level == AxeleraLevel::Core) {
            schedule.set_property("num_cores", std::to_string(cores));
        }
        // Iterations are independent and need no synchronization, so CPU code generators may run them in order.
        structured_control_flow::ScheduleType_Serializable::set(schedule);
        return schedule;
    }

    static bool is(const structured_control_flow::ScheduleType& schedule, AxeleraLevel level) {
        if (schedule.value() != value()) {
            return false;
        }
        auto it = schedule.properties().find("level");
        return it != schedule.properties().end() && it->second == to_string(level);
    }

    static int64_t num_cores(const structured_control_flow::ScheduleType& schedule) {
        return std::stoll(schedule.properties().at("num_cores"));
    }
};

} // namespace axelera
} // namespace sdfg
