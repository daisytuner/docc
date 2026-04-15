/**
 * @file data_flow_node.h
 * @brief Base class for all nodes in the dataflow graph
 *
 * DataFlowNode is the abstract base class for all nodes that can appear in a dataflow
 * graph. It provides the fundamental interface for nodes including vertex management,
 * parent graph access, validation, and cloning.
 *
 * ## Key Concepts
 *
 * ### Node Hierarchy
 * The node hierarchy is:
 * - DataFlowNode (abstract base)
 *   - AccessNode: Data access points (variables, arrays)
 *     - ConstantNode: Constant literal values
 *   - CodeNode: Computational operations (abstract)
 *     - Tasklet: Simple operations (add, mul, etc.)
 *     - LibraryNode: Complex operations (BLAS, etc.)
 *
 * ### Graph Vertex
 * Each node is associated with a Boost graph vertex that represents its position
 * in the dataflow graph. The vertex is used for graph traversal and edge management.
 *
 * ### Parent Graph
 * Each node maintains a reference to its parent DataFlowGraph, which owns the node
 * and manages its lifetime and connections.
 *
 * ## Example Usage
 *
 * Working with nodes through the base interface:
 * @code
 * // Get the node's vertex for graph operations
 * graph::Vertex v = node.vertex();
 *
 * // Get the parent dataflow graph
 * DataFlowGraph& graph = node.get_parent();
 *
 * // Check incoming/outgoing edges
 * size_t in_degree = graph.in_degree(node);
 * size_t out_degree = graph.out_degree(node);
 *
 * // Validate the node
 * node.validate(function);
 * @endcode
 *
 * @see AccessNode for data access nodes
 * @see CodeNode for computational nodes
 * @see DataFlowGraph for the container graph
 * @see Element for the base element class
 */

#pragma once

#include <boost/lexical_cast.hpp>
#include <nlohmann/json.hpp>

#include "sdfg/element.h"
#include "sdfg/graph/graph.h"

using json = nlohmann::json;

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

enum class EdgeRemoveOption {
    /** You must not remove this edge unless you can guarantee the entire node will be removed **/
    NotRemovable,
    /**
     * You may remove the edge without further action
     **/
    Trivially,
    /**
     * It can be removed, but the node requires a call to update itself to match the removed edge
     **/
    RequiresUpdate,
    /**
     * This is the only relevant edge of this node. If it will be removed, the entire node must be removed as well,
     * ignoring any potential side-effect flags of the node
     **/
    RemoveNodeAfter
};

class DataFlowGraph;
class Memlet;

/**
 * @class DataFlowNode
 * @brief Abstract base class for all dataflow graph nodes
 *
 * DataFlowNode provides the core interface for all nodes in the dataflow graph.
 * Key responsibilities:
 * - Vertex management: Each node has a unique graph vertex
 * - Parent graph access: Nodes know their containing graph
 * - Validation: Abstract interface for semantic validation
 * - Cloning: Abstract interface for creating node copies
 *
 * This is an abstract class and cannot be instantiated directly.
 * Use derived classes like AccessNode, Tasklet, or LibraryNode.
 */
class DataFlowNode : public Element {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    // Remark: Exclusive resource
    graph::Vertex vertex_; ///< Graph vertex representing this node's position

    DataFlowGraph* parent_; ///< Parent dataflow graph that owns this node

protected:
    /**
     * @brief Protected constructor for dataflow nodes
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this node
     * @param vertex Graph vertex for this node
     * @param parent Parent dataflow graph
     */
    DataFlowNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent);

public:
    // Remark: Exclusive resource
    DataFlowNode(const DataFlowNode& data_node) = delete;
    DataFlowNode& operator=(const DataFlowNode&) = delete;

    /**
     * @brief Check if this node has side effects
     * Side effect on dflow nodes is broad. It just means it cannot be removed just because there are no more direct
     * consumers in the dataflow But underneath, a accessNode-write is considered as having a side effect for this
     * purpose. Closer inspection of access nodes could show those side effects as being irrelevant if never read and
     * fully owned
     */
    [[nodiscard]] virtual bool side_effect() const = 0;

    /**
     * @brief Get the graph vertex for this node
     * @return The Boost graph vertex representing this node
     */
    graph::Vertex vertex() const;

    /**
     * @brief Get the parent dataflow graph (const)
     * @return Const reference to the parent DataFlowGraph
     */
    const DataFlowGraph& get_parent() const;

    /**
     * @brief Get the parent dataflow graph (mutable)
     * @return Mutable reference to the parent DataFlowGraph
     */
    DataFlowGraph& get_parent();

    /**
     * @brief Clone this node for graph transformations
     * @param element_id New element identifier for the clone
     * @param vertex New graph vertex for the clone
     * @param parent Parent graph for the clone
     * @return Unique pointer to the cloned node
     *
     * Pure virtual function that must be implemented by derived classes
     * to support graph transformations and optimizations.
     */
    virtual std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const = 0;

    /**
     * Is the edge in question removable, or will this make the node its an output on invalid?
     * Currently, some nodes may require every out connector to have at least 1 connection attached
     * @return NotRemovable, NoDependencies, CustomRemove
     *
     * [NotRemovable] means only if the entire node can be removed, can the edge be removed
     * [NoDependencies] means, the edge can be removed trivially, the node will work without for the remaining functions
     * [CustomRemove] means, a call to [remove_out_edge] can have the node remove the edge itself and make necessary
     */
    virtual EdgeRemoveOption can_remove_out_edge(const data_flow::DataFlowGraph& graph, const Memlet* memlet) const = 0;

    /**
     *
     * @param out_conn a output connector, whose edge was just removed after approval via [can_remove_out_edge] ==
     * EdgeRemoveOption::RequiresUpdate
     * @return edge removal is completed, node is valid without edge
     */
    virtual bool update_edge_removed(const std::string& out_conn) { return false; }
};
} // namespace data_flow
} // namespace sdfg
