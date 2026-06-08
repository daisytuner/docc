/**
 * @file matmul_node.h
 * @brief Tensor matrix multiplication node compatible with ONNX MatMul operator
 *
 * This file defines the MatMulNode class which implements a tensor matrix
 * multiplication operation following the ONNX MatMul operator specification.
 *
 * ## ONNX MatMul Operator Compatibility
 *
 * The MatMulNode implements the ONNX MatMul operator with the following semantics:
 * - Input tensor A: [..., M, K] - arbitrary batch dimensions followed by matrix dims
 * - Input tensor B: [..., K, N] - arbitrary batch dimensions followed by matrix dims
 * - Output tensor Y: [..., M, N] - broadcasted batch dimensions with result matrix
 *
 * The operation performs matrix multiplication on the last two dimensions and
 * broadcasts over the batch dimensions following numpy broadcasting rules.
 *
 * ## Expansion
 *
 * The matmul operation is expanded into nested maps:
 * 1. Create outer maps for parallel iteration over batch and output dimensions
 * 2. Create inner loop for sequential accumulation over the K dimension
 * 3. Compute matrix multiplication using FMA (fused multiply-add) operations
 * 4. Write results to output tensor
 *
 * ## Example
 *
 * Creating a batched matrix multiplication:
 * @code
 * symbolic::MultiExpression shape_a = {symbolic::symbol("B"), symbolic::symbol("M"), symbolic::symbol("K")};
 * symbolic::MultiExpression shape_b = {symbolic::symbol("B"), symbolic::symbol("K"), symbolic::symbol("N")};
 *
 * auto& matmul_node = builder.add_library_node<math::tensor::MatMulNode>(
 *     block, debug_info, shape_a, shape_b
 * );
 * @endcode
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_MatMul("ml::MatMul");

/**
 * @class MatMulNode
 * @brief Tensor matrix multiplication following ONNX MatMul operator specification
 *
 * MatMulNode represents a tensor matrix multiplication operation that is compatible
 * with the ONNX MatMul operator. The operation performs matrix multiplication on
 * the last two dimensions and supports broadcasting over batch dimensions.
 *
 * ## Input/Output Requirements
 * - Input connector "A": Input tensor [..., M, K]
 * - Input connector "B": Input tensor [..., K, N]
 * - Output connector "Y": Output tensor [..., M, N]
 *
 * ## Broadcasting
 * The batch dimensions are broadcast following numpy broadcasting rules:
 * - Dimensions are compared from right to left (excluding last two matrix dims)
 * - Dimensions must be equal, or one of them must be 1
 * - The output shape takes the maximum of each dimension
 *
 * ## Example
 *
 * For inputs A[B, M, K] and B[B, K, N]:
 * - Y = A @ B has shape [B, M, N]
 * - Each (b, m, n) element is computed as: sum_k(A[b, m, k] * B[b, k, n])
 */
class MatMulNode : public TensorNode {
private:
    types::PrimitiveType fixed_quantization_;
    TensorLayout layout_a_;
    TensorLayout layout_b_;
    // TensorLayout layout_y_; // we are not using that yet

    /** @deprecated use TensorLayout **/
    static bool has_basic_strides(symbolic::MultiExpression shape, symbolic::MultiExpression strides);
    /** @deprecated use TensorLayout **/
    static bool has_transposed_strides(symbolic::MultiExpression shape, symbolic::MultiExpression strides);

public:
    /**
     * @brief Construct a matmul node
     * @param element_id Unique element identifier
     * @param debug_info Debug information
     * @param vertex Graph vertex
     * @param parent Parent dataflow graph
     */
    MatMulNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const TensorLayout& layout_a,
        const TensorLayout& layout_b,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static auto constexpr Y_INPUT_IDX = 0;
    static auto constexpr A_INPUT_IDX = 1;
    static auto constexpr B_INPUT_IDX = 2;


    /**
     * @brief Get the M dimension (rows of A, rows of output)
     * @return M dimension expression
     */
    symbolic::Expression m() const;

    /**
     * @brief Get the N dimension (columns of B, columns of output)
     * @return N dimension expression
     */
    symbolic::Expression n() const;

    QuantizationType quantization() const { return quantization(get_parent()); }

    /**
     * type of the math calculations. May be inferred or fixed.
     */
    QuantizationType quantization(const data_flow::DataFlowGraph& dataflow) const;

    /**
     * Same result as quantization if it matches all the inputs. None if its impossible to use the same types
     * for input & output and math
     */
    std::optional<QuantizationType> uniform_quantization(const data_flow::DataFlowGraph& dataflow) const;

    /**
     * configuration of the type for the math calculations, independent of current input types etc.
     * 'Void' indicates auto-inferring from inputs
     */
    QuantizationType fixed_quantization() const;

    void set_fixed_quantization(const QuantizationType quant);

    const TensorLayout& layout_a() const;

    const TensorLayout& layout_b() const;

    /**
     * @brief Get the K dimension (columns of A, rows of B - contraction dimension)
     * @return K dimension expression
     */
    symbolic::Expression k() const;

    void validate(const Function& function) const override;

    /**
     * @brief Expand matmul into nested maps
     *
     * Expands the matmul operation by:
     * 1. Creating outer maps for parallel iteration over batch and M, N dimensions
     * 2. Creating inner for loop for sequential accumulation over K dimension
     * 3. Computing matrix multiplication using FMA (fused multiply-add) tasklets
     * 4. Writing results to output tensor
     *
     * @param builder SDFG builder
     * @param analysis_manager Analysis manager
     * @return True if expansion succeeded
     */
    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    symbolic::Expression flop() const override;

    bool supports_integer_types() const override { return true; }
};

/**
 * @class MatMulNodeSerializer
 * @brief Serializer for MatMulNode
 */
class MatMulNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
