#pragma once

#include <memory>

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
 * @brief Represents a traditional for-loop with sequential execution
 *
 * A For loop is a structured loop that executes sequentially with:
 * - Initialization: Sets the induction variable to its initial value
 * - Condition: Evaluated before each iteration; loop continues while true
 * - Update: Updates the induction variable after each iteration
 * - Body: Sequence of control flow nodes executed each iteration
 *
 * **Example:**
 * ```cpp
 * for (int i = 0; i < 10; i++) {
 *   // body
 * }
 * ```
 *
 * Corresponds to:
 * - indvar: i
 * - init: 0
 * - condition: i < 10
 * - update: i + 1
 *
 * @see StructuredLoop
 * @see Map
 */
class For : public StructuredLoop {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    For(size_t element_id,
        const DebugInfo& debug_info,
        ControlFlowNode* parent,
        symbolic::Symbol indvar,
        symbolic::Expression init,
        symbolic::Expression update,
        symbolic::Condition condition);

    static constexpr size_t REQUIRED_ELEMENT_IDS = StructuredLoop::REQUIRED_ELEMENT_IDS;

public:
    For(const For& node) = delete;
    For& operator=(const For&) = delete;

    void validate(const Function& function) const override;
};

} // namespace structured_control_flow
} // namespace sdfg
