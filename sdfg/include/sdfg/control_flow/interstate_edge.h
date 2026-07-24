#pragma once

#include <boost/lexical_cast.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "sdfg/control_flow/state.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

using json = nlohmann::json;

namespace sdfg {

namespace builder {
class SDFGBuilder;
}

namespace control_flow {

/**
 * @brief Type alias for symbol assignments on interstate edges.
 *
 * Assignments map symbols (left-hand side) to symbolic expressions (right-hand side).
 * Assignments must involve integer scalar symbols only.
 */
typedef symbolic::SymbolMap Assignments;

/**
 * @brief An edge connecting two states in an SDFG control-flow graph.
 *
 * InterstateEdge represents a transition between states in a Stateful DataFlow Graph.
 * Each edge can have:
 * - A **condition**: A boolean expression that guards the transition
 * - **Assignments**: Updates to symbols that occur during the transition
 *
 * ## Execution Semantics
 *
 * **IMPORTANT**: When an interstate edge is traversed, the execution order is:
 * 1. **First**, the condition is evaluated using the current symbol values
 * 2. **Then**, if the condition is true, the assignments are executed to update symbols
 * 3. Finally, control transfers to the destination state
 *
 * This means:
 * - The condition sees symbol values BEFORE any assignments on this edge
 * - Assignments cannot affect the condition evaluation on the same edge
 * - Assignments affect symbol values for subsequent edges and states
 *
 * ## Examples
 *
 * @code
 * // Example 1: Loop counter increment
 * // Condition uses current value of i, then i is incremented
 * symbolic::Symbol i = symbolic::symbol("i");
 * auto condition = symbolic::Lt(i, symbolic::integer(10));  // Check i < 10
 * Assignments assignments;
 * assignments[i] = symbolic::add(i, symbolic::integer(1));  // Then i = i + 1
 *
 * // Example 2: Unconditional transition with assignment
 * auto condition = symbolic::__true__();
 * Assignments assignments2;
 * assignments2[i] = symbolic::integer(0);  // Initialize i = 0
 * @endcode
 *
 * @see State for the source and destination of transitions
 * @see symbolic::Condition for the condition expressions
 */
class InterstateEdge : public Element {
    friend class builder::SDFGBuilder;

private:
    // Remark: Exclusive resource
    const graph::Edge edge_;

    const control_flow::State& src_;
    const control_flow::State& dst_;

    symbolic::Condition condition_;
    control_flow::Assignments assignments_;

    InterstateEdge(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Edge& edge,
        const control_flow::State& src,
        const control_flow::State& dst,
        const symbolic::Condition condition,
        const control_flow::Assignments& assignments
    );

public:
    // Remark: Exclusive resource
    InterstateEdge(const InterstateEdge& state) = delete;
    InterstateEdge& operator=(const InterstateEdge&) = delete;

    ElementType type_id() const override { return ElementType::InterstateEdge; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::InterstateEdge; }

    /**
     * @brief Validates the interstate edge within the context of a function.
     *
     * Validates that:
     * - The condition is a valid boolean expression
     * - All symbols in the condition and assignments are properly typed
     * - Assignment targets are integer scalar types
     * - Assignment sources evaluate to integer or pointer types
     *
     * @param function The function containing this edge
     * @throws InvalidSDFGException if validation fails
     */
    void validate(const Function& function) const override;

    /**
     * @brief Returns the graph edge representing this interstate edge.
     * @return The graph edge
     */
    const graph::Edge edge() const;

    /**
     * @brief Returns the source state of this edge.
     * @return Reference to the source state
     */
    const control_flow::State& src() const;

    /**
     * @brief Returns the destination state of this edge.
     * @return Reference to the destination state
     */
    const control_flow::State& dst() const;

    /**
     * @brief Returns the condition guarding this transition.
     *
     * The condition is evaluated BEFORE any assignments are executed.
     * The condition uses the symbol values as they exist when entering the edge.
     *
     * @return The boolean condition expression
     * @see is_unconditional() to check if this is an unconditional transition
     */
    const symbolic::Condition condition() const;

    /**
     * @brief Checks if this edge represents an unconditional transition.
     * @return true if the condition is always true, false otherwise
     */
    bool is_unconditional() const;

    /**
     * @brief Returns the symbol assignments for this edge.
     *
     * Assignments are executed AFTER the condition is evaluated and found to be true.
     * The assignments update symbol values that will be visible in the destination state
     * and subsequent edges.
     *
     * @return Map of symbols to their assigned expressions
     */
    const control_flow::Assignments& assignments() const;

    /**
     * @brief Replaces symbolic expressions in the edge's condition and assignments.
     * @param old_expression The expression to replace
     * @param new_expression The replacement expression
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

} // namespace control_flow
} // namespace sdfg
