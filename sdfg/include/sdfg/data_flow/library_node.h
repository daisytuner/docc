/**
 * @file library_node.h
 * @brief Library node definitions for calling functions beyond simple instructions
 *
 * Library nodes are objects of the dataflow graph that represent pre-defined operations
 * that can have multiple implementations and can be expanded into more primitive operations.
 *
 * ## Key Concepts
 *
 * ### Library Nodes
 * Library nodes provide a high-level abstraction for complex operations such as:
 * - Mathematical operations (BLAS, tensor operations)
 * - Standard library functions (malloc, memcpy, memset)
 * - Custom function calls
 *
 * ### Implementation Types
 * Library nodes can have different implementations specified via ImplementationType:
 * - ImplementationType_NONE: No specific implementation, single dispatcher or expansion
 * - Custom implementation types (e.g., BLAS, CUBLAS) for selection of dispatcher
 *
 * The implementation type determines how the library node is code-generated:
 * - If implementation_type is NONE, the node may only have a single dispatcher or must be expanded
 * - If implementation_type is set, a dispatcher generates specific code
 *
 * ## Example
 *
 * Creating and using a library node:
 * @code
 * // Library node with BLAS implementation
 * auto& gemm_node = builder.add_library_node<math::blas::GemmNode>(
 *     block, debug_info, implementation_type, precision, layout, ...
 * );
 *
 * // Library node that will be expanded
 * auto& tensor_add = builder.add_library_node<math::tensor::AddNode>(
 *     block, debug_info, shape
 * );
 * @endcode
 *
 * @see math::MathNode for mathematical library nodes
 * @see codegen::LibraryNodeDispatcher for code generation
 * @see serializer::LibraryNodeSerializer for serialization
 */

#pragma once

#include <string>
#include <vector>

#include "sdfg/data_flow/code_node.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {

namespace builder {
class SDFGBuilder;
class StructuredSDFGBuilder;
} // namespace builder

namespace data_flow {

typedef StringEnum LibraryNodeCode;

/**
 * @brief Implementation type identifier for library nodes
 *
 * Determines how a library node should be implemented during code generation.
 * Each library node can specify an implementation type that indicates:
 * - Whether is only has a single dispatcher or expand the node into primitive operations (NONE)
 * - Which library implementation to use (BLAS, CUBLAS, etc.)
 */
typedef StringEnum ImplementationType;

/**
 * @brief Default implementation type indicating no specific implementation
 *
 * When a library node has ImplementationType_NONE, it may be code-generated
 * using a single dispatcher or is expanded into more primitive operations
 * during the expansion pass.
 */
inline ImplementationType ImplementationType_NONE{""};

/**
 * @class LibraryNode
 * @brief Base class for all library nodes in the dataflow graph
 *
 * Library nodes represent calls to functions beyond simple instructions (tasklets).
 * They can be mathematical operations, standard library functions, or custom operations.
 * Each library node has:
 * - A code identifying the operation (e.g., "GEMM", "Add", "Memcpy")
 * - An implementation type specifying how to generate code
 * - Optional side effects
 *
 * Derived classes implement specific operations and may provide expansion methods
 * to convert high-level operations into primitive operations.
 */
class LibraryNode : public CodeNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

protected:
    LibraryNodeCode code_; ///< Operation identifier (e.g., "GEMM", "Add")
    bool side_effect_; ///< Whether this node has side effects

    ImplementationType implementation_type_; ///< How to implement this node

    /**
     * @brief Protected constructor for library nodes
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this node
     * @param vertex Graph vertex for this node
     * @param parent Parent dataflow graph
     * @param code Operation code/identifier
     * @param outputs Output connector names
     * @param inputs Input connector names
     * @param side_effect Whether this operation has side effects
     * @param implementation_type Implementation strategy
     */
    LibraryNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const LibraryNodeCode& code,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        const bool side_effect,
        const ImplementationType& implementation_type
    );

public:
    LibraryNode(const LibraryNode& data_node) = delete;
    LibraryNode& operator=(const LibraryNode&) = delete;

    virtual ~LibraryNode() = default;

    /**
     * @brief Get the operation code
     * @return Library node code identifier
     */
    const LibraryNodeCode& code() const;

    /**
     * @brief Get the implementation type (const)
     * @return Implementation type for this node
     */
    const ImplementationType& implementation_type() const;

    /**
     * @brief Get the implementation type (mutable)
     * @return Mutable reference to implementation type
     */
    ImplementationType& implementation_type();

    /**
     * @brief Check if this node has side effects
     * @return True if the node has side effects
     */
    [[nodiscard]] bool side_effect() const override;

    /**
     * @brief Convert node to string representation
     * @return String representation of the node
     */
    virtual std::string toStr() const;

    /**
     * @brief Get all symbols used in this node
     * @return Set of symbolic expressions used by this node
     */
    virtual symbolic::SymbolSet symbols() const = 0;

    /**
     * @brief Calculate floating point operations for this node
     * @return Symbolic expression for FLOP count (or null)
     */
    virtual symbolic::Expression flop() const;

    /**
     * Describes what a pointer is used for
     * @param input_idx index of input that is a pointer.
     * @return Invalid if not asked about a pointer input
     */
    virtual PointerAccessType pointer_access_type(int input_idx) const { return PointerUnknownAccess(); }

    EdgeRemoveOption can_remove_out_edge(const data_flow::DataFlowGraph& graph, const Memlet* memlet) const override;
};

} // namespace data_flow
} // namespace sdfg
