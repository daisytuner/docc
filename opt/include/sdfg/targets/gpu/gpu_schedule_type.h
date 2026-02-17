#pragma once

#include <string>

#include "sdfg/exceptions.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/targets/gpu/gpu_types.h"

namespace sdfg {
namespace gpu {

/**
 * @brief Base class for GPU schedule types (CUDA/HIP) using CRTP pattern
 *
 * This template class provides shared functionality for both CUDA and HIP
 * schedule types. Derived classes only need to implement:
 * - static const std::string value() - Returns "CUDA" or "HIP"
 * - static symbolic::Integer default_block_size_x() - Default block size for X dimension
 *
 * @tparam Derived The derived class (ScheduleType_CUDA or ScheduleType_HIP)
 */
template<typename Derived>
class ScheduleType_GPU_Base {
public:
    /**
     * @brief Set the GPU dimension for a schedule
     */
    static void dimension(structured_control_flow::ScheduleType& schedule, const GPUDimension& dimension) {
        schedule.set_property("dimension", std::to_string(dimension));
    }

    /**
     * @brief Get the GPU dimension from a schedule
     */
    static GPUDimension dimension(const structured_control_flow::ScheduleType& schedule) {
        return static_cast<GPUDimension>(std::stoi(schedule.properties().at("dimension")));
    }

    /**
     * @brief Set the block size for a schedule
     */
    static void block_size(structured_control_flow::ScheduleType& schedule, const symbolic::Expression block_size) {
        serializer::JSONSerializer serializer;
        schedule.set_property("block_size", serializer.expression(block_size));
    }

    /**
     * @brief Get the block size from a schedule
     * Returns default values if not explicitly set:
     * - X: Derived::default_block_size_x() (32 for CUDA, 64 for HIP)
     * - Y: 8
     * - Z: 4
     */
    static symbolic::Integer block_size(const structured_control_flow::ScheduleType& schedule) {
        if (schedule.properties().find("block_size") == schedule.properties().end()) {
            if (dimension(schedule) == GPUDimension::X) {
                return Derived::default_block_size_x();
            } else if (dimension(schedule) == GPUDimension::Y) {
                return symbolic::integer(8);
            } else if (dimension(schedule) == GPUDimension::Z) {
                return symbolic::integer(4);
            } else {
                throw InvalidSDFGException("Invalid GPU dimension");
            }
        }
        std::string expr_str = schedule.properties().at("block_size");
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
    static structured_control_flow::ScheduleType create() {
        auto schedule_type = structured_control_flow::
            ScheduleType(Derived::value(), structured_control_flow::ScheduleTypeCategory::Offloader);
        dimension(schedule_type, GPUDimension::X);
        return schedule_type;
    }
};

} // namespace gpu
} // namespace sdfg
