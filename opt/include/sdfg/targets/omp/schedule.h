#pragma once

#include <string>

#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace omp {

/**
 * @brief OpenMP scheduling strategies
 */
enum OpenMPSchedule {
    Static, ///< Iterations distributed statically at compile time
    Dynamic, ///< Iterations distributed dynamically at runtime
    Guided ///< Chunk sizes decrease over time (dynamic with feedback)
};

/**
 * @brief CPU parallel schedule type with OpenMP support
 *
 * Indicates that loop iterations can execute in parallel on CPU threads
 * using OpenMP directives. Supports configuration of:
 * - Number of threads
 * - OpenMP scheduling strategy (static, dynamic, guided)
 */
class ScheduleType_OMP {
public:
    /**
     * @brief Set the number of threads for parallel execution
     * @param schedule Schedule to configure
     * @param num_threads Symbolic expression for number of threads
     */
    static void num_threads(structured_control_flow::ScheduleType& schedule, const symbolic::Expression num_threads) {
        serializer::JSONSerializer serializer;
        schedule.set_property("num_threads", serializer.expression(num_threads));
    }

    /**
     * @brief Get the number of threads
     * @param schedule Schedule to query
     * @return Symbolic expression for number of threads
     */
    static const symbolic::Expression num_threads(const structured_control_flow::ScheduleType& schedule) {
        serializer::JSONSerializer serializer;
        if (schedule.properties().find("num_threads") == schedule.properties().end()) {
            return SymEngine::null;
        }
        std::string expr_str = schedule.properties().at("num_threads");
        auto expr = symbolic::parse(expr_str);
        return expr;
    }

    /**
     * @brief Set the OpenMP scheduling strategy
     * @param schedule Schedule to configure
     * @param schedule_type OpenMP scheduling strategy
     */
    static void omp_schedule(structured_control_flow::ScheduleType& schedule, const OpenMPSchedule schedule_type) {
        schedule.set_property("omp_schedule", std::to_string(schedule_type));
    }

    /**
     * @brief Get the OpenMP scheduling strategy
     * @param schedule Schedule to query
     * @return OpenMP scheduling strategy
     */
    static OpenMPSchedule omp_schedule(const structured_control_flow::ScheduleType& schedule) {
        if (schedule.properties().find("omp_schedule") == schedule.properties().end()) {
            return OpenMPSchedule::Static;
        }
        return static_cast<OpenMPSchedule>(std::stoi(schedule.properties().at("omp_schedule")));
    }

    static const std::string value() { return "CPU_PARALLEL"; }

    static structured_control_flow::ScheduleType create() {
        return structured_control_flow::ScheduleType(value(), structured_control_flow::ScheduleTypeCategory::Parallelizer);
    }
};

} // namespace omp
} // namespace sdfg
