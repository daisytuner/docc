/**
 * @file access_node.h
 * @brief Access nodes for reading/writing data in the dataflow graph
 *
 * Access nodes represent data access points in the SDFG dataflow graph. They serve as
 * interfaces between the dataflow graph and the function's data containers. Access nodes
 * can represent reads from or writes to variables, arrays, and other data structures.
 *
 * ## Key Concepts
 *
 * ### AccessNode
 * An AccessNode represents a reference to a data container in the function. Multiple
 * access nodes can reference the same data container, representing different access points
 * in the dataflow. Access nodes are connected to code nodes (tasklets, library nodes) via
 * memlets that describe the data movement.
 *
 * ### ConstantNode
 * A ConstantNode is a special type of access node that represents a constant literal value.
 * Unlike regular access nodes, constant nodes:
 * - Do not reference function variables
 * - Cannot have incoming edges
 * - Carry their own type information
 * - Are validated to ensure the literal matches the type
 *
 * ## Example Usage
 *
 * Creating access nodes for reading and writing:
 * @code
 * // Add data containers
 * types::Scalar float_type(types::PrimitiveType::Float);
 * builder.add_container("input", float_type);
 * builder.add_container("output", float_type);
 *
 * // Create access nodes
 * auto& read_access = builder.add_access(state, "input");
 * auto& write_access = builder.add_access(state, "output");
 *
 * // Connect via a tasklet
 * auto& tasklet = builder.add_tasklet(state, TaskletCode::assign, "_out", {"_in"});
 * builder.add_computational_memlet(state, read_access, tasklet, "_in", {});
 * builder.add_computational_memlet(state, tasklet, "_out", write_access, {});
 * @endcode
 *
 * Creating a constant node:
 * @code
 * types::Scalar int_type(types::PrimitiveType::Int32);
 * auto& constant = builder.add_constant(state, "42", int_type);
 * @endcode
 *
 * @see DataFlowNode for the base class
 * @see Memlet for edge connections
 * @see CodeNode for computational nodes
 */

#pragma once

#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/graph/graph.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

/**
 * @class AccessNode
 * @brief Node representing access to a data container in the dataflow graph
 *
 * AccessNode provides an interface for reading from or writing to data containers
 * defined in the function. Each access node references a data container by name and
 * can be connected to code nodes via memlets.
 *
 * Access nodes support:
 * - Reference to function data containers
 * - Multiple accesses to the same container
 * - Validation of data container existence
 * - Symbol replacement for container renaming
 * - Cloning for graph transformations
 */
class AccessNode : public DataFlowNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

protected:
    std::string data_; ///< Name of the data container this node accesses

    /**
     * @brief Protected constructor for access nodes
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this node
     * @param vertex Graph vertex for this node
     * @param parent Parent dataflow graph
     * @param data Name of the data container to access
     */
    AccessNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const std::string& data
    );

public:
    AccessNode(const AccessNode& data_node) = delete;
    AccessNode& operator=(const AccessNode&) = delete;

    /**
     * @brief Validate the access node
     * @param function Function context for validation
     * @throws InvalidSDFGException if validation fails
     *
     * Validates that:
     * - The referenced data container exists in the function
     * - All outgoing edges have the same memlet type
     * - All incoming edges have the same memlet type
     */
    void validate(const Function& function) const override;

    /**
     * @brief Get the name of the data container
     * @return Name of the accessed data container
     */
    const std::string& data() const;

    /**
     * @brief Set the name of the data container
     * @param data New name for the data container
     */
    void data(const std::string data);

    /**
     * An access node with incoming edges writes the container and therefore has side effects
     */
    [[nodiscard]] bool side_effect() const override;

    /**
     * @brief Clone this access node for graph transformations
     * @param element_id New element identifier for the clone
     * @param vertex New graph vertex for the clone
     * @param parent Parent graph for the clone
     * @return Unique pointer to the cloned node
     */
    virtual std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;

    /**
     * @brief Replace symbolic expressions in this node
     * @param old_expression Expression to replace
     * @param new_expression Replacement expression
     *
     * If both expressions are symbols and old_expression matches the data name,
     * replaces the data name with new_expression.
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    EdgeRemoveOption can_remove_out_edge(const DataFlowGraph& graph, const Memlet* memlet) const override;
};

/**
 * @class ConstantNode
 * @brief Node representing a constant literal value in the dataflow graph
 *
 * ConstantNode is a specialized access node that represents constant literals
 * (e.g., "42", "3.14", "true"). Unlike regular access nodes:
 * - Does not reference a function variable
 * - Cannot have incoming edges (constants are sources only)
 * - Stores its own type information
 * - Validates that the literal value matches the declared type
 *
 * Constants are useful for:
 * - Providing literal values to computations
 * - Initializing variables
 * - Specifying algorithm parameters
 */
class ConstantNode : public AccessNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    std::unique_ptr<types::IType> type_; ///< Type of the constant value

    /**
     * @brief Protected constructor for constant nodes
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this node
     * @param vertex Graph vertex for this node
     * @param parent Parent dataflow graph
     * @param data String representation of the constant value
     * @param type Type of the constant
     */
    ConstantNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const std::string& data,
        const types::IType& type
    );

public:
    ConstantNode(const ConstantNode& data_node) = delete;
    ConstantNode& operator=(const ConstantNode&) = delete;

    /**
     * @brief Get the type of this constant
     * @return Type of the constant value
     */
    const types::IType& type() const;

    [[nodiscard]] bool side_effect() const override { return false; }

    /**
     * @brief Validate the constant node
     * @param function Function context for validation
     * @throws InvalidSDFGException if validation fails
     *
     * Validates that:
     * - The data name does not conflict with function variables
     * - The node has no incoming edges
     * - For integer/boolean types, the literal is a valid number
     */
    void validate(const Function& function) const override;

    /**
     * @brief Clone this constant node for graph transformations
     * @param element_id New element identifier for the clone
     * @param vertex New graph vertex for the clone
     * @param parent Parent graph for the clone
     * @return Unique pointer to the cloned node
     */
    virtual std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;
};


} // namespace data_flow
} // namespace sdfg
