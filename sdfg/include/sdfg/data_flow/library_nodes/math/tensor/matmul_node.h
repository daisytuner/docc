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

#include <optional>

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
 *
 * ## Element types
 * A and B share the input type; the products are summed in the accumulator type and stored in the output type
 * (see ContractionTypes), e.g. Int8 x Int8 accumulated and stored as Int32.
 */
class MatMulNode : public TensorNode {
private:
    ContractionTypes types_;
    TensorLayout layout_a_;
    TensorLayout layout_b_;
    TensorLayout layout_y_;

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
        const TensorLayout* layout_y = nullptr,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    /**
     * @brief Construct a matmul node with explicit element types
     * @param types input, accumulator and output types; Void entries are inferred
     */
    MatMulNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const TensorLayout& layout_a,
        const TensorLayout& layout_b,
        const ContractionTypes& types,
        const TensorLayout* layout_y = nullptr,
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

    /// The configured element types; Void entries are inferred, see input_type(), accumulator_type(), output_type().
    const ContractionTypes& types() const;

    void set_types(const ContractionTypes& types);

    /// Element type of A and B: configured, or inferred from their memlets (Void if they disagree).
    types::PrimitiveType input_type() const;

    /// Element type the products are summed in: configured, or the output type.
    types::PrimitiveType accumulator_type() const;

    /// Element type of Y: configured, or inferred from its memlet.
    types::PrimitiveType output_type() const;

    /**
     * The single type of inputs, accumulator and output, if they all agree. None if the matmul mixes types, such as
     * Int8 inputs into an Int32 output.
     */
    std::optional<types::PrimitiveType> uniform_quantization(const data_flow::DataFlowGraph& dataflow) const;

    [[deprecated("Use input_type(), accumulator_type(), output_type() or uniform_quantization() instead")]]
    QuantizationType quantization() const {
        return quantization(get_parent());
    }

    [[deprecated("Use input_type(), accumulator_type(), output_type() or uniform_quantization() instead")]]
    QuantizationType quantization(const data_flow::DataFlowGraph& dataflow) const;

    [[deprecated("Use input_type() instead")]]
    types::PrimitiveType input_quantization() const;

    [[deprecated("Use output_type() instead")]]
    types::PrimitiveType output_quantization() const;

    /// The configured type if inputs and output share one (Void otherwise).
    [[deprecated("Use types() instead")]]
    QuantizationType fixed_quantization() const;

    [[deprecated("Use set_types(ContractionTypes::uniform(...)) instead")]]
    void set_fixed_quantization(const QuantizationType quant);

    const TensorLayout& layout_a() const;

    const TensorLayout& layout_b() const;

    const TensorLayout& layout_y() const;

    TensorLayout& layout_a() {
        return layout_a_;
    }

    TensorLayout& layout_b() {
        return layout_b_;
    }

    TensorLayout& layout_y() {
        return layout_y_;
    }

    static TensorLayout get_linear_result_layout(const TensorLayout& layout_a, const TensorLayout& layout_b);

    /**
     * @brief Get the K dimension (columns of A, rows of B - contraction dimension)
     * @return K dimension expression
     */
    symbolic::Expression k() const;

    void validate(const Function& function) const override;

    void verify_data_types(const data_flow::DataFlowGraph& graph) const override;

    /**
     * @brief Expand matmul into nested maps
     *
     * Expands the matmul operation by:
     * 1. Creating outer maps for parallel iteration over batch and M, N dimensions
     * 2. Creating inner for loop for sequential accumulation over K dimension
     * 3. Computing matrix multiplication using FMA (fused multiply-add) tasklets
     * 4. Writing results to output tensor
     *
     * @param context way to request isolation of node or add new blocks, loops etc.
     * @param block the block in which the node is contained
     * @return outcome of expand, created by context
     */
    virtual passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext&, structured_control_flow::Block& block) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    symbolic::Expression flop() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    bool relocalize_operand(int input_idx, const TensorLayout& packed) override;
    bool can_relocalize_operand(int input_idx, const TensorLayout& packed) const override;

    bool supports_integer_types() const override {
        return true;
    }
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

class MatMulNaiveExpander : public passes::CodeLibNodeExpander<MatMulNode> {
public:
    MatMulNaiveExpander() : passes::CodeLibNodeExpander<MatMulNode>(LibraryNodeType_MatMul) {
    }

    passes::LibNodeExpander::ExpandOutcome handle_expand(
        passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, MatMulNode& node
    ) const override;
};

class MatMulBlasExpander : public passes::CodeLibNodeExpander<MatMulNode> {
private:
    bool supports_half_; //< Not all BLAS implementations support half precision (hgemm)

public:
    MatMulBlasExpander(bool supports_half = false)
        : passes::CodeLibNodeExpander<MatMulNode>(LibraryNodeType_MatMul), supports_half_(supports_half) {
    }


    passes::LibNodeExpander::ExpandOutcome handle_expand(
        passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, MatMulNode& node
    ) const override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
