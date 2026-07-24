/**
 * @file code_node.h
 * @brief Base class for computational nodes in the dataflow graph
 *
 * Code nodes represent computational operations in the SDFG dataflow graph. They are
 * abstract nodes that have inputs and outputs connected via memlets. Code nodes include
 * tasklets (simple operations) and library nodes (complex operations).
 *
 * ## Key Concepts
 *
 * ### Code Node
 * A CodeNode is the base class for all computational nodes. It defines:
 * - Input connectors: Named inputs that receive data via memlets
 * - Output connectors: Named outputs that send data via memlets
 * - Connection interface for memlets
 *
 * ### Input/Output Connectors
 * Connectors are named strings that identify specific inputs and outputs.
 * For example, a binary operation might have inputs "_in1" and "_in2" and
 * output "_out". Memlets connect access nodes to these connectors.
 *
 * ### Derived Classes
 * - Tasklet: Simple operations like addition, multiplication, assignment
 * - LibraryNode: Complex operations like BLAS calls, reductions, custom functions
 *
 * ## Example Usage
 *
 * Creating a code node (via Tasklet):
 * @code
 * // Create a binary addition tasklet
 * auto& tasklet = builder.add_tasklet(
 *     state,
 *     TaskletCode::fp_add,
 *     "_out",              // output connector
 *     {"_in1", "_in2"}     // input connectors
 * );
 *
 * // Connect inputs and output
 * builder.add_computational_memlet(state, a_node, tasklet, "_in1", {});
 * builder.add_computational_memlet(state, b_node, tasklet, "_in2", {});
 * builder.add_computational_memlet(state, tasklet, "_out", result_node, {});
 * @endcode
 *
 * Accessing inputs and outputs:
 * @code
 * const auto& inputs = tasklet.inputs();   // Get all input connector names
 * const auto& outputs = tasklet.outputs(); // Get all output connector names
 * std::string first_input = tasklet.input(0);  // Get first input by index
 * bool is_const = tasklet.has_constant_input(0); // Check if input is constant
 * @endcode
 *
 * @see DataFlowNode for the base class
 * @see Tasklet for simple operations
 * @see LibraryNode for complex operations
 * @see Memlet for edge connections
 */

#pragma once

#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/graph/graph.h"
#include "sdfg/types/scalar.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

/**
 * @class CodeNode
 * @brief Base class for computational nodes with inputs and outputs
 *
 * CodeNode represents any computational operation in the dataflow graph.
 * It provides:
 * - Named input and output connectors
 * - Methods to query and modify connectors
 * - Detection of constant inputs
 * - Abstract cloning interface for graph transformations
 *
 * Derived classes must implement:
 * - validate(): Validate the node's configuration
 * - clone(): Create a copy of the node
 */
class CodeNode : public DataFlowNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

protected:
    std::vector<std::string> outputs_; ///< Names of output connectors
    std::vector<std::string> inputs_; ///< Names of input connectors

    /**
     * @brief Protected constructor for code nodes
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this node
     * @param vertex Graph vertex for this node
     * @param parent Parent dataflow graph
     * @param outputs Names of output connectors
     * @param inputs Names of input connectors
     */
    CodeNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs
    );

public:
    /// Bitmask of all concrete ElementTypes that are CodeNodes.
    static constexpr ElementType TypeGroup = ElementType::Tasklet | ElementType::LibraryNode;

    /// LLVM-style RTTI predicate: true if \p element is a CodeNode.
    static bool classof(const Element& element) { return is_a(element.type_id(), TypeGroup); }

    CodeNode(const CodeNode& data_node) = delete;
    CodeNode& operator=(const CodeNode&) = delete;

    /**
     * @brief Validate the code node's configuration
     * @param function Function context for validation
     *
     * Ensures that:
     * - No two access nodes connect to the same data for inputs/outputs
     */
    void validate(const Function& function) const override;

    /**
     * @brief Get output connector names (const)
     * @return Const reference to vector of output connector names
     */
    const std::vector<std::string>& outputs() const;

    /**
     * @brief Get input connector names (const)
     * @return Const reference to vector of input connector names
     */
    const std::vector<std::string>& inputs() const;

    /**
     * @brief Get output connector names (mutable)
     * @return Mutable reference to vector of output connector names
     */
    std::vector<std::string>& outputs();

    /**
     * @brief Get input connector names (mutable)
     * @return Mutable reference to vector of input connector names
     */
    std::vector<std::string>& inputs();

    /**
     * @brief Get output connector name by index
     * @param index Index of the output connector
     * @return Name of the output connector at the given index
     */
    const std::string& output(size_t index) const;

    /**
     * @brief Get input connector name by index
     * @param index Index of the input connector
     * @return Name of the input connector at the given index
     */
    const std::string& input(size_t index) const;

    /**
     * @brief Check if an input is connected to a constant node
     * @param index Index of the input connector to check
     * @return True if the input is connected to a ConstantNode, false otherwise
     */
    bool has_constant_input(size_t index) const;
};
} // namespace data_flow
} // namespace sdfg
