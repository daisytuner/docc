#pragma once

#include <memory>

#include "sdfg/control_flow/interstate_edge.h"
#include "sdfg/control_flow/state.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class StructuredSDFGBuilder;
}

namespace structured_control_flow {

/**
 * @brief A control flow node representing a basic block with dataflow computations
 *
 * A Block is the fundamental computation unit in a StructuredSDFG. It represents
 * a basic block that contains a dataflow graph describing the computations to be
 * performed. The dataflow graph consists of nodes (tasklets, access nodes, library
 * nodes) and memlets (data movement edges).
 *
 * Blocks are analogous to States in the unstructured SDFG model, but they exist
 * within the structured control flow hierarchy. Each Block contains:
 * - A DataFlowGraph describing the computations and data movements
 * - Implicit sequential ordering with respect to other control flow nodes
 *
 * Blocks can contain:
 * - Tasklets: Small code snippets performing computations
 * - Access nodes: Read/write access to containers (arrays, scalars)
 * - Library nodes: Calls to library functions (BLAS, tensor operations, etc.)
 * - Memlets: Edges describing data movement between nodes
 *
 * @see data_flow::DataFlowGraph
 * @see data_flow::Tasklet
 * @see control_flow::State
 */
class Block : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    std::unique_ptr<data_flow::DataFlowGraph> dataflow_;

    Block(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent);

    static constexpr size_t REQUIRED_ELEMENT_IDS = 1;

public:
    Block(const Block& block) = delete;
    Block& operator=(const Block&) = delete;

    ElementType type_id() const override { return ElementType::Block; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::Block; }

    bool accept(visitor::ActualStructuredSDFGVisitor& visitor) override;

    void validate(const Function& function) const override;

    /**
     * @brief Access the dataflow graph (const version)
     * @return Const reference to the dataflow graph
     */
    const data_flow::DataFlowGraph& dataflow() const;

    /**
     * @brief Access the dataflow graph (non-const version)
     * @return Reference to the dataflow graph for modification
     */
    data_flow::DataFlowGraph& dataflow();

    /**
     * @brief Replace occurrences of an expression in the dataflow graph
     * @param old_expression Expression to replace
     * @param new_expression Expression to replace with
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

    template<typename T>
    T* is_a_library_node() {
        return this->dataflow().is_a_library_node<T>();
    }
};

class AssignmentBlock : public ControlFlowNode {
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    control_flow::Assignments assignments_;

    AssignmentBlock(
        ElementId element_id,
        const DebugInfo& debug_info,
        ControlFlowNode* parent,
        const control_flow::Assignments& assignments
    );
    AssignmentBlock(ElementId element_id, const DebugInfo& debug_info, ControlFlowNode* parent);

    static constexpr size_t REQUIRED_ELEMENT_IDS = 1;

public:
    AssignmentBlock(const AssignmentBlock& block) = delete;
    AssignmentBlock& operator=(const AssignmentBlock&) = delete;

    ElementType type_id() const override { return ElementType::AssignmentBlock; }

    /**
     * Add this assignment, if the symbol in question is not already being written.
     * In other words this is a "add_before". Sync all assignments are evaluated simultaneously, if it is already
     * written, the new assignment is already dead code.
     * @return true if it was added, the symbol was not already written to in this block
     */
    bool add_if_not_overwritten(const symbolic::Symbol& target, const symbolic::Expression& expr);

    static bool classof(const Element& element) { return element.type_id() == ElementType::AssignmentBlock; }

    bool accept(visitor::ActualStructuredSDFGVisitor& visitor) override;

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
     * @brief Replace occurrences of an expression in the dataflow graph
     * @param old_expression Expression to replace
     * @param new_expression Expression to replace with
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};

} // namespace structured_control_flow
} // namespace sdfg
