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
 * @brief Represents a transition between control flow nodes in a sequence
 *
 * A Transition connects consecutive control flow nodes within a Sequence.
 * It can contain assignments that update symbol values when the transition
 * is taken. Transitions are similar to InterstateEdge in the unstructured
 * SDFG model but exist within the structured control flow hierarchy.
 *
 * Each control flow node in a Sequence has an associated Transition that
 * is executed after the node completes. The transition's assignments update
 * symbol values for subsequent nodes.
 *
 * @see Sequence
 * @see control_flow::InterstateEdge
 */
class Transition : public Element {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    Sequence* parent_;
    control_flow::Assignments assignments_;

    Transition(size_t element_id, const DebugInfo& debug_info, Sequence& parent);

    Transition(
        size_t element_id, const DebugInfo& debug_info, Sequence& parent, const control_flow::Assignments& assignments
    );

public:
    Transition(const Transition& node) = delete;
    Transition& operator=(const Transition&) = delete;

    void validate(const Function& function) const override;

    /**
     * @brief Access the assignments in this transition (const version)
     * @return Const reference to the assignments map
     */
    const control_flow::Assignments& assignments() const;

    /**
     * @brief Access the assignments in this transition (non-const version)
     * @return Reference to the assignments map for modification
     */
    control_flow::Assignments& assignments();

    /**
     * @brief Get the parent sequence (non-const version)
     * @return Reference to the parent sequence
     */
    Sequence& parent();

    /**
     * @brief Get the parent sequence (const version)
     * @return Const reference to the parent sequence
     */
    const Sequence& parent() const;

    /**
     * @brief Check if this transition has no assignments
     * @return true if assignments map is empty, false otherwise
     */
    bool empty() const;

    /**
     * @brief Get the number of assignments in this transition
     * @return Number of symbol assignments
     */
    size_t size() const;

    /**
     * @brief Replace occurrences of an expression in assignments
     * @param old_expression Expression to replace
     * @param new_expression Expression to replace with
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    using Element::replace;
};

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
    std::vector<std::unique_ptr<Transition>> transitions_;

    Sequence(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent);

    static constexpr size_t REQUIRED_ELEMENT_IDS = 1;

public:
    Sequence(const Sequence& node) = delete;
    Sequence& operator=(const Sequence&) = delete;

    void validate(const Function& function) const override;

    /**
     * @brief Get the number of children in this sequence
     * @return Number of child control flow nodes
     */
    size_t size() const;

    /**
     * @brief Access a child and its transition by index (const version)
     * @param i Index of the child to access (0-based)
     * @return Pair of (child node, transition)
     * @throws std::out_of_range if i >= size()
     */
    std::pair<const ControlFlowNode&, const Transition&> at(size_t i) const;

    /**
     * @brief Access a child and its transition by index (non-const version)
     * @param i Index of the child to access (0-based)
     * @return Pair of (child node, transition)
     * @throws std::out_of_range if i >= size()
     */
    std::pair<ControlFlowNode&, Transition&> at(size_t i);

    /**
     * @brief Find the index of a child node
     * @param child Child node to search for
     * @return Index of the child, or -1 if not found
     */
    int index(const ControlFlowNode& child) const;

    /**
     * @brief Find the index of a transition
     * @param transition Transition to search for
     * @return Index of the transition, or -1 if not found
     */
    int index(const Transition& transition) const;

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
