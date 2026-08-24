#pragma once

#include <string>

#include "sdfg/exceptions.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/targets/gpu/gpu_types.h"

namespace sdfg {
namespace gpu {

enum class TargetLevel {
    X_GRID,
    Y_GRID,
    Z_GRID,
    X_BLOCK,
    Y_BLOCK,
    Z_BLOCK,
    WARP,
};

inline std::string to_string(const TargetLevel& target_level) {
    switch (target_level) {
        case TargetLevel::X_GRID:
            return "X_GRID";
        case TargetLevel::Y_GRID:
            return "Y_GRID";
        case TargetLevel::Z_GRID:
            return "Z_GRID";
        case TargetLevel::X_BLOCK:
            return "X_BLOCK";
        case TargetLevel::Y_BLOCK:
            return "Y_BLOCK";
        case TargetLevel::Z_BLOCK:
            return "Z_BLOCK";
        case TargetLevel::WARP:
            return "WARP";
    }
    throw InvalidSDFGException("Invalid TargetLevel");
}

inline TargetLevel target_level_from_string(const std::string& value) {
    if (value == "X_GRID") {
        return TargetLevel::X_GRID;
    } else if (value == "Y_GRID") {
        return TargetLevel::Y_GRID;
    } else if (value == "Z_GRID") {
        return TargetLevel::Z_GRID;
    } else if (value == "X_BLOCK") {
        return TargetLevel::X_BLOCK;
    } else if (value == "Y_BLOCK") {
        return TargetLevel::Y_BLOCK;
    } else if (value == "Z_BLOCK") {
        return TargetLevel::Z_BLOCK;
    } else if (value == "WARP") {
        return TargetLevel::WARP;
    }
    throw InvalidSDFGException("Invalid TargetLevel: " + value);
}

/**
 * @brief Base class for GPU schedule types (CUDA/ROCm) using CRTP pattern
 *
 * This template class provides shared functionality for both CUDA and ROCm
 * schedule types. Derived classes only need to implement:
 * - static const std::string value() - Returns "CUDA" or "ROCm"
 * - static symbolic::Integer default_block_size_x() - Default block size for X dimension
 *
 * @tparam Derived The derived class (ScheduleType_CUDA or ScheduleType_ROCM)
 */

class ScheduleType_GPU_Offload {
public:
    /**
     * @brief Set the target level for a schedule
     */
    static void target_level(structured_control_flow::ScheduleType& schedule, const TargetLevel& target_level) {
        schedule.set_property("target_level", gpu::to_string(target_level));
    }

    /**
     * @brief Get the target level from a schedule
     */
    static TargetLevel target_level(const structured_control_flow::ScheduleType& schedule) {
        return gpu::target_level_from_string(schedule.properties().at("target_level"));
    }

    /**
     * @brief Set the parallel size for a schedule
     */
    static void parallel_size(structured_control_flow::ScheduleType& schedule, const symbolic::Integer parallel_size) {
        serializer::JSONSerializer serializer;
        schedule.set_property("parallel_size", serializer.expression(parallel_size));
    }

    /**
     * @brief Get the parallel size from a schedule
     */
    static symbolic::Integer parallel_size(const structured_control_flow::ScheduleType& schedule) {
        if (schedule.properties().find("parallel_size") == schedule.properties().end()) {
            throw InvalidSDFGException("Parallel size not set for schedule type: " + schedule.value());
        }
        std::string expr_str = schedule.properties().at("parallel_size");
        return symbolic::integer(std::stoi(expr_str));
    }

    /**
     * @brief Check if nested synchronization is enabled
     */
    static bool nested_sync(const structured_control_flow::ScheduleType& schedule) {
        if (schedule.properties().find("nested_sync") == schedule.properties().end()) {
            return false;
        }
        std::string val = schedule.properties().at("nested_sync");
        return val == "true";
    }

    /**
     * @brief Set nested synchronization flag
     */
    static void nested_sync(structured_control_flow::ScheduleType& schedule, const bool nested_sync) {
        schedule.set_property("nested_sync", nested_sync ? "true" : "false");
    }

    /**
     * @brief Create a new GPU schedule type
     */
    template<typename Derived>
    static structured_control_flow::ScheduleType
    create(const TargetLevel& target_level_, const symbolic::Integer& parallel_size_) {
        auto schedule_type = structured_control_flow::
            ScheduleType(Derived::value(), structured_control_flow::ScheduleTypeCategory::Offloader);
        target_level(schedule_type, target_level_);
        parallel_size(schedule_type, parallel_size_);
        return schedule_type;
    }
};

/**
 * @brief Get the GPU target level from any GPU schedule type
 */
inline TargetLevel gpu_target_level(const structured_control_flow::ScheduleType& schedule) {
    return target_level_from_string(schedule.properties().at("target_level"));
}

/**
 * @brief Set the GPU target level on any GPU schedule type
 */
inline void gpu_target_level(structured_control_flow::ScheduleType& schedule, const TargetLevel& target_level) {
    schedule.set_property("target_level", to_string(target_level));
}

} // namespace gpu
} // namespace sdfg
