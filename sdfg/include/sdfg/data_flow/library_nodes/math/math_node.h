/**
 * @file math_node.h
 * @brief Mathematical library node base class with expansion support
 *
 * This file defines the MathNode base class for all mathematical library nodes.
 * Math nodes support expansion, which converts high-level mathematical operations
 * into primitive SDFG operations (maps, tasklets, etc.).
 *
 * ## Expansion
 *
 * Expansion is the process of transforming a library node into more primitive
 * operations. For example:
 * - A tensor elementwise addition can be expanded into a map with a tasklet
 * - A reduction operation can be expanded into nested maps with accumulation
 *
 * The expansion process is controlled by:
 * - The implementation_type field: If set to ImplementationType_NONE, expansion is attempted
 * - The expand() method: Subclasses implement the expansion logic
 *
 * ## Example
 *
 * Expanding a tensor operation:
 * @code
 * auto& tensor_add = builder.add_library_node<math::tensor::AddNode>(block, debug_info, shape);
 *
 * analysis::AnalysisManager analysis_manager(sdfg);
 * if (tensor_add.expand(builder, analysis_manager)) {
 *     // Node was successfully expanded into primitive operations
 * }
 * @endcode
 *
 * @see math::tensor::ElementWiseUnaryNode for tensor elementwise operations
 * @see math::tensor::ReduceNode for tensor reduction operations
 * @see math::blas::BLASNode for BLAS operations
 */

#pragma once

#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {

namespace analysis {
class AnalysisManager;
}

namespace builder {
class StructuredSDFGBuilder;
}

namespace math {

/**
 * @class MathNode
 * @brief Base class for mathematical library nodes with expansion support
 *
 * MathNode extends LibraryNode with the ability to expand high-level mathematical
 * operations into primitive SDFG operations. Subclasses implement specific
 * mathematical operations (tensor operations, BLAS operations, etc.) and provide
 * expansion methods when appropriate.
 *
 * All math nodes can optionally be expanded during the expansion pass if their
 * implementation_type is set to ImplementationType_NONE.
 */
class MathNode : public data_flow::LibraryNode {
public:
    /**
     * @brief Construct a mathematical library node
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this node
     * @param vertex Graph vertex for this node
     * @param parent Parent dataflow graph
     * @param code Operation code/identifier
     * @param outputs Output connector names
     * @param inputs Input connector names
     * @param implementation_type Implementation strategy
     */
    MathNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        const data_flow::ImplementationType& implementation_type,
        bool side_effects = false
    );

    /**
     * @brief Expand this node into primitive operations
     *
     * The expansion process transforms the high-level mathematical operation
     * represented by this node into primitive SDFG constructs (maps, tasklets, etc.).
     * This allows the operation to be further optimized and scheduled.
     *
     * @param builder SDFG builder for creating new nodes
     * @param analysis_manager Analysis manager for querying SDFG properties
     * @return True if expansion was successful, false otherwise
     */
    virtual bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) = 0;
};

} // namespace math
} // namespace sdfg
