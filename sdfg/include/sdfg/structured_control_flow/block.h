#pragma once

#include <memory>

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

    Block(size_t element_id, const DebugInfo& debug_info);

public:
    Block(const Block& block) = delete;
    Block& operator=(const Block&) = delete;

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

    template<typename T>
    T* is_a_library_node() {
        return this->dataflow().is_a_library_node<T>();
    }
};

} // namespace structured_control_flow
} // namespace sdfg
