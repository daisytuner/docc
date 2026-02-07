#pragma once

#include <memory>
#include <unordered_map>

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}
namespace structured_control_flow {

/**
 * @brief Categories of schedule types for Map nodes
 * Defines the high-level classification of scheduling strategies.
 */
enum class ScheduleTypeCategory { Offloader, Parallelizer, Vectorizer, None };

/**
 * @brief Represents a schedule type for Map nodes
 *
 * ScheduleType encapsulates the execution schedule for Map loops, including
 * the scheduling strategy and associated properties. Different schedule types
 * control how loop iterations are distributed and executed.
 */
class ScheduleType {
private:
    std::unordered_map<std::string, std::string> properties_;
    std::string value_;
    ScheduleTypeCategory category_;

public:
    ScheduleType(std::string value, ScheduleTypeCategory category) : value_(value), category_(category) {}

    /**
     * @brief Get the schedule type identifier
     * @return Schedule type string (e.g., "SEQUENTIAL", "CPU_PARALLEL")
     */
    const std::string& value() const { return value_; }

    /**
     * @brief Get all schedule properties
     * @return Map of property names to values
     */
    const std::unordered_map<std::string, std::string>& properties() const { return properties_; }

    /**
     * @brief Get the schedule type category
     * @return Schedule type category enum
     */
    ScheduleTypeCategory category() const { return category_; }

    /**
     * @brief Set a schedule property
     * @param key Property name
     * @param value Property value
     */
    void set_property(const std::string& key, const std::string& value) {
        if (properties_.find(key) == properties_.end()) {
            properties_.insert({key, value});
            return;
        }
        properties_.at(key) = value;
    }

    void operator=(const ScheduleType& rhs) {
        value_ = rhs.value_;
        properties_.clear();
        for (const auto& entry : rhs.properties_) {
            properties_.insert(entry);
        }
    }
};


/**
 * @brief Sequential schedule type for Map nodes
 *
 * Indicates that loop iterations execute sequentially in order.
 */
class ScheduleType_Sequential {
public:
    static const std::string value() { return "SEQUENTIAL"; }
    static ScheduleType create() { return ScheduleType(value(), ScheduleTypeCategory::None); }
};

/**
 * @brief Represents a parallel map loop with configurable scheduling
 *
 * A Map is a special type of structured loop that can be executed in parallel.
 * Unlike For loops which execute sequentially, Map loops explicitly indicate
 * that iterations are independent and can be distributed across parallel
 * execution units (threads, cores, etc.).
 *
 * Maps support different scheduling strategies:
 * - Sequential: Iterations execute sequentially (debugging, baseline)
 * - CPU Parallel: Iterations execute in parallel using OpenMP on CPU
 * - GPU: Iterations map to GPU execution (future extension)
 *
 * **Example:**
 * ```cpp
 * map (int i = 0; i < N; i++) {
 *   C[i] = A[i] + B[i];  // Iterations are independent
 * }
 * ```
 *
 * Maps are commonly used for:
 * - Data-parallel operations (element-wise array operations)
 * - Loop nests that can be parallelized
 * - Performance-critical sections requiring parallel execution
 *
 * @see StructuredLoop
 * @see For
 * @see ScheduleType
 */
class Map : public StructuredLoop {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    ScheduleType schedule_type_;

    Map(size_t element_id,
        const DebugInfo& debug_info,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition,
        const ScheduleType& schedule_type);

public:
    Map(const Map& node) = delete;
    Map& operator=(const Map&) = delete;

    void validate(const Function& function) const override;

    /**
     * @brief Get the scheduling strategy for this Map
     * @return The schedule type (sequential, parallel, etc.)
     */
    const ScheduleType& schedule_type() const;
};

} // namespace structured_control_flow
} // namespace sdfg
