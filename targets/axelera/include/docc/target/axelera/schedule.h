#pragma once

#include <sdfg/structured_control_flow/structured_loop.h>

namespace sdfg {
namespace axelera {

/// Lanes of a PWORD, the vector unit every Axelera stream moves per beat.
inline constexpr size_t PWORD_LANES = 64;

/**
 * Loop over PWORD blocks that becomes `A.grid` around one datapath; its iterations step by PWORD_LANES elements.
 * Its body is a single ScheduleType_Axelera_Lane loop.
 */
class ScheduleType_Axelera_Datapath {
public:
    static const std::string value() {
        return "AXELERA_DATAPATH";
    }

    static structured_control_flow::ScheduleType create() {
        return {value(), structured_control_flow::ScheduleTypeCategory::Offloader};
    }
};

/// Loop over the PWORD_LANES lanes of one block; the units process all lanes at once.
class ScheduleType_Axelera_Lane {
public:
    static const std::string value() {
        return "AXELERA_LANE";
    }

    static structured_control_flow::ScheduleType create() {
        return {value(), structured_control_flow::ScheduleTypeCategory::Vectorizer};
    }
};

} // namespace axelera
} // namespace sdfg
