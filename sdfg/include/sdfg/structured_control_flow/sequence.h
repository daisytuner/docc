#pragma once

#include <memory>

#include "sdfg/control_flow/interstate_edge.h"
#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {

class StructuredSDFG;

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

class While;
class StructuredLoop;
class Sequence;

/**
 * @brief A sequential container of control flow nodes
 *
 * A Sequence represents a sequential execution of control flow nodes. It is
 * the fundamental container in structured control flow, serving as:
 * - The root container of a StructuredSDFG
 * - The body of loops (For, While, Map)
 * - Each branch of an IfElse
 *
 * A Sequence contains:
 * - A list of child control flow nodes (Block, IfElse, loops, etc.)
 * - A transition for each child (containing symbol assignments)
 *
 * Children are executed sequentially in order. After each child completes,
 * its associated transition executes, potentially updating symbol values
 * before the next child begins.
 *
 * **Structure:**
 * ```
 * Sequence:
 *   Child[0] -> Transition[0] (assignments)
 *   Child[1] -> Transition[1] (assignments)
 *   ...
 *   Child[n-1] -> Transition[n-1] (assignments)
 * ```
 *
 * @see ControlFlowNode
 * @see Transition
 * @see StructuredSDFG::root()
 */
class Sequence : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

    friend class sdfg::StructuredSDFG;

    friend class sdfg::structured_control_flow::While;
    friend class sdfg::structured_control_flow::StructuredLoop;

private:
    std::vector<std::unique_ptr<ControlFlowNode>> children_;

    Sequence(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent);

    static constexpr size_t REQUIRED_ELEMENT_IDS = 1;

public:
    Sequence(const Sequence& node) = delete;
    Sequence& operator=(const Sequence&) = delete;

    ElementType type_id() const override { return ElementType::Sequence; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::Sequence; }

    bool accept(visitor::ActualStructuredSDFGVisitor& visitor) override;

    void validate(const Function& function) const override;

    /**
     * @brief Get the number of children in this sequence
     * @return Number of child control flow nodes
     */
    size_t size() const;

    /**
     * @brief Access a child by index (const version)
     * @param i Index of the child to access (0-based)
     * @return child node
     * @throws std::out_of_range if i >= size()
     */
    const ControlFlowNode& at(size_t i) const;

    /**
     * @brief Access a child by index (non-const version)
     * @param i Index of the child to access (0-based)
     * @return child node
     * @throws std::out_of_range if i >= size()
     */
    ControlFlowNode& at(size_t i);

    /**
     * @brief Find the index of a child node
     * @param child Child node to search for
     * @return Index of the child, or -1 if not found
     */
    int index(const ControlFlowNode& child) const;

    /**
     * @brief Replace occurrences of an expression in all children and transitions
     * @param old_expression Expression to replace
     * @param new_expression Expression to replace with
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};

} // namespace structured_control_flow
} // namespace sdfg
