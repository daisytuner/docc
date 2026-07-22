/**
 * @file memlet.h
 * @brief Edges in the dataflow graph representing data movement
 *
 * Memlets are edges in the SDFG dataflow graph that describe data movement between
 * nodes. They specify what data is transferred, how it is accessed, and the type of
 * the data being moved.
 *
 * ## Key Concepts
 *
 * ### Memlet Types
 * Memlets can be of several types:
 * - **Computational**: Transfers data between access nodes and code nodes (most common)
 * - **Reference**: Creates a pointer/reference to data
 * - **Dereference_Src/Dereference_Dst**: Dereferences pointers
 *
 * ### Connectors
 * Memlets connect specific connectors on nodes:
 * - **void connector**: Used on access nodes (no named connector)
 * - **named connectors**: Used on code nodes (e.g., "_in1", "_out")
 *
 * ### Subset
 * A subset describes which portion of data is accessed, represented as a vector of
 * symbolic expressions. For scalars, the subset is empty. For arrays, it specifies
 * indices or ranges.
 *
 * ### Base Type
 * The base type is the type of the data container being accessed. The actual data
 * transferred may be a scalar element (via subset) or the full container.
 *
 * ## Example Usage
 *
 * Creating a computational memlet:
 * @code
 * // Transfer scalar data from access node to tasklet input
 * builder.add_computational_memlet(
 *     state,
 *     access_node,  // source
 *     tasklet,      // destination
 *     "_in",        // destination connector
 *     {}            // empty subset for scalar
 * );
 * @endcode
 *
 * Creating a memlet with array subset:
 * @code
 * // Access element at index i
 * auto i = symbolic::symbol("i");
 * builder.add_computational_memlet(
 *     state,
 *     array_access,
 *     tasklet,
 *     "_in",
 *     {i}  // subset: array[i]
 * );
 * @endcode
 *
 * Querying memlet properties:
 * @code
 * MemletType type = memlet.type();
 * const DataFlowNode& src = memlet.src();
 * const DataFlowNode& dst = memlet.dst();
 * const std::string& conn = memlet.dst_conn();
 * const Subset& subset = memlet.subset();
 * @endcode
 *
 * @see DataFlowNode for node types
 * @see AccessNode for data access points
 * @see CodeNode for computation nodes
 * @see DataFlowGraph for graph container
 */

#pragma once

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

class DataFlowGraph;

/**
 * @typedef Subset
 * @brief Vector of symbolic expressions describing data access pattern
 *
 * A subset specifies which portion of a data container is accessed.
 * Each dimension is represented by a symbolic expression.
 * Examples:
 * - Empty vector: Full scalar or container access
 * - {i}: Single element at index i
 * - {i, j}: 2D array element at [i][j]
 */
typedef std::vector<symbolic::Expression> Subset;

/**
 * @enum MemletType
 * @brief Type of data movement represented by a memlet
 */
enum MemletType {
    Computational, ///< Standard data transfer for computation
    Reference, ///< Creates a reference/pointer to data
    Dereference_Src, ///< Dereferences pointer at source
    Dereference_Dst ///< Dereferences pointer at destination
};

/**
 * @class Memlet
 * @brief Edge in the dataflow graph representing data movement
 *
 * Memlets are the edges of the dataflow graph that describe how data flows
 * between nodes. Each memlet specifies:
 * - Source and destination nodes
 * - Source and destination connectors
 * - Data subset being transferred
 * - Base type of the data
 * - Type of data movement
 *
 * Memlets are validated to ensure correct connections between node types
 * and proper type usage.
 */
class Memlet : public Element {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    // Remark: Exclusive resource
    const graph::Edge edge_; ///< Graph edge representing this memlet

    DataFlowGraph* parent_; ///< Parent dataflow graph

    DataFlowNode& src_; ///< Source node
    DataFlowNode& dst_; ///< Destination node
    std::string src_conn_; ///< Source connector name
    std::string dst_conn_; ///< Destination connector name
    Subset subset_; ///< Data access subset
    std::unique_ptr<types::IType> base_type_; ///< Base type of the data

    /**
     * @brief Constructor for memlets
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this memlet
     * @param edge Graph edge for this memlet
     * @param parent Parent dataflow graph
     * @param src Source node
     * @param src_conn Source connector name ("void" for access nodes)
     * @param dst Destination node
     * @param dst_conn Destination connector name ("void" for access nodes)
     * @param subset Data access subset
     * @param base_type Base type of the data being transferred
     */
    Memlet(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Edge& edge,
        DataFlowGraph& parent,
        DataFlowNode& src,
        const std::string& src_conn,
        DataFlowNode& dst,
        const std::string& dst_conn,
        const Subset& subset,
        const types::IType& base_type
    );

public:
    // Remark: Exclusive resource
    Memlet(const Memlet& memlet) = delete;
    Memlet& operator=(const Memlet&) = delete;

    ElementType type_id() const override { return ElementType::Memlet; }

    static bool classof(const Element& element) { return element.type_id() == ElementType::Memlet; }

    /**
     * @brief Validate the memlet
     * @param function Function context for validation
     * @throws InvalidSDFGException if validation fails
     *
     * Validates that:
     * - Subset dimensions are not null
     * - For computational memlets: connects code and access nodes correctly
     * - For computational memlets: connector names are valid
     * - For computational memlets to tasklets: result type is scalar
     * - For reference memlets: destination is an access node with pointer type
     */
    void validate(const Function& function) const override;

    /**
     * @brief Get the graph edge for this memlet
     * @return The Boost graph edge
     */
    const graph::Edge edge() const;

    /**
     * @brief Get the parent dataflow graph (const)
     * @return Const reference to parent graph
     */
    const DataFlowGraph& get_parent() const;

    /**
     * @brief Get the parent dataflow graph (mutable)
     * @return Mutable reference to parent graph
     */
    DataFlowGraph& get_parent();

    /**
     * @brief Determine the type of this memlet
     * @return MemletType indicating the kind of data movement
     */
    MemletType type() const;

    /**
     * Memlet actually reads src (from memory). Address calculations with the src as symbol are not reads.
     * If src is local memory,
     */
    bool is_src_read() const;

    /**
     * Memlet returns the contents of src (this is [is_src_read()], excluding that the contents are only used
     * transitively)
     */
    bool is_src_direct_read() const;

    /**
     * Makes a memory read with src-contents as (partial) address
     */
    bool is_src_pointed_to_read() const;

    /**
     * Leaks the address of src or uses it in an address calculation that is returned. If the result is not DeadCode,
     * somebody else could use it to change src, without using the container
     */
    bool is_src_address_leak() const;

    /**
     * Leaks address of what src points to. Only valid for when src is a pointer-like
     */
    bool is_src_pointed_to_address_leak(const types::IType& src_type) const;

    /**
     * makes a memory write to dst (itself). Using dst as pointer is not a write to dst
     */
    bool is_dst_write() const;

    /**
     * Contents of dst are used in the address calculation of a memory write
     */
    bool is_dst_pointed_to_write() const;

    /**
     * Contents of dst are read
     */
    bool is_dst_read() const;

    /**
     * @brief Get the source node (const)
     * @return Const reference to source node
     */
    const DataFlowNode& src() const;

    /**
     * @brief Get the source node (mutable)
     * @return Mutable reference to source node
     */
    DataFlowNode& src();

    /**
     * @brief Get the destination node (const)
     * @return Const reference to destination node
     */
    const DataFlowNode& dst() const;

    /**
     * @brief Get the destination node (mutable)
     * @return Mutable reference to destination node
     */
    DataFlowNode& dst();

    /**
     * @brief Get the source connector name
     * @return Source connector name ("void" for access nodes)
     */
    const std::string& src_conn() const;

    /**
     * @brief Get the destination connector name
     * @return Destination connector name ("void" for access nodes)
     */
    const std::string& dst_conn() const;

    /**
     * @brief Get the data access subset
     * @return Vector of symbolic expressions describing the subset
     */
    const Subset& subset() const;

    /**
     * @brief Set the data access subset
     * @param subset New subset vector
     */
    void set_subset(const Subset& subset);

    /**
     * @brief Get the base type of the data
     * @return Type of the data container being accessed
     */
    const types::IType& base_type() const;

    /**
     * @brief Set the base type of the data
     * @param base_type New base type
     */
    void set_base_type(const types::IType& base_type);

    /**
     * @brief Compute the result type after applying subset
     * @param function Function context for type inference
     * @return The inferred result type
     */
    std::unique_ptr<types::IType> result_type(const Function& function) const;

    /**
     * @brief Clone this memlet for graph transformations
     * @param element_id New element identifier
     * @param edge New graph edge
     * @param parent New parent graph
     * @param src New source node
     * @param dst New destination node
     * @return Unique pointer to cloned memlet
     */
    std::unique_ptr<Memlet> clone(
        size_t element_id, const graph::Edge& edge, DataFlowGraph& parent, DataFlowNode& src, DataFlowNode& dst
    ) const;

    /**
     * @brief Replace symbolic expressions in this memlet
     * @param old_expression Expression to replace
     * @param new_expression Replacement expression
     *
     * Replaces occurrences in the subset vector & the base type.
     */
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;
};
} // namespace data_flow
} // namespace sdfg
