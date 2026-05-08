/**
 * @file tensor_node.h
 * @brief Abstract base class for tensor library nodes
 *
 * This file defines the abstract base class TensorNode which provides common
 * validation and type determination functionality for all tensor operations.
 *
 * ## Type Requirements
 *
 * Tensor nodes require that:
 * - All memlets are scalar or pointer of scalar types
 * - All connected memlets have the same primitive type (e.g., all Float or all Int32)
 * - The primitive type is consistent across inputs and outputs
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

constexpr types::PrimitiveType QUANTIZATION_MATCH_INPUTS = types::PrimitiveType::Void;

/**
 * @class TensorNode
 * @brief Abstract base class for all tensor operations
 *
 * TensorNode provides common functionality for tensor library nodes including:
 * - Validation that all memlets are scalar or pointer of scalar
 * - Validation that all memlets have the same primitive type
 * - Method to determine the primitive type of the operation
 *
 * Derived classes must specify whether they support integer types via
 * supports_integer_types().
 */
class TensorNode : public math::MathNode {
public:
    /**
     * @brief Construct a tensor node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     * @param code Operation code
     * @param outputs Output connector names
     * @param inputs Input connector names
     * @param impl_type Implementation type
     */
    TensorNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        data_flow::ImplementationType impl_type
    );

    /**
     * @brief Validate tensor node constraints
     *
     * Validates that:
     * - All input/output memlets are scalar or pointer of scalar
     * - All memlets have the same primitive type
     * - If the operation doesn't support integers, the type is floating-point
     *
     * @param function The function containing this node
     * @throws InvalidSDFGException if validation fails
     */
    void validate(const Function& function) const override;

    /**
     * @brief Get the primitive type of this tensor operation
     *
     * Determines the primitive type by examining connected memlets.
     * All memlets must have the same primitive type.
     *
     * TODO this may not be true for all operations and will require a filter for which inputs to check
     *
     * @param graph The dataflow graph containing this node
     * @return The primitive type used by this operation
     * @throws InvalidSDFGException if types are inconsistent
     */
    types::PrimitiveType primitive_type(const data_flow::DataFlowGraph& graph) const;

    /**
     * @brief Check if this operation supports integer types
     *
     * Operations like add, sub, mul, div, min, max support both floating-point
     * and integer types. Operations like exp, log, tanh, sigmoid only support
     * floating-point types.
     *
     * @return true if integer types are supported, false otherwise
     */
    virtual bool supports_integer_types() const = 0;

protected:
    /**
     * @brief Get the appropriate tasklet code for min/max operations on integers
     *
     * Selects between signed and unsigned variants based on the primitive type.
     *
     * @param prim_type The primitive type
     * @param is_max True for max operation, false for min operation
     * @return The appropriate tasklet code
     */
    static data_flow::TaskletCode get_integer_minmax_tasklet(types::PrimitiveType prim_type, bool is_max);
};

} // namespace tensor
} // namespace math
} // namespace sdfg
