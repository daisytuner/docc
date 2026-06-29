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
    Map(size_t element_id,
        const DebugInfo& debug_info,
        ControlFlowNode* parent,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition,
        const ScheduleType& schedule_type);

    static constexpr size_t REQUIRED_ELEMENT_IDS = StructuredLoop::REQUIRED_ELEMENT_IDS;

public:
    Map(const Map& node) = delete;
    Map& operator=(const Map&) = delete;

    void validate(const Function& function) const override;
};

} // namespace structured_control_flow
} // namespace sdfg
