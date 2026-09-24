#pragma once

#include <optional>
#include <string>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Attention("ml::Attention");

/**
 * @class AttentionNode
 * @brief Fused scaled-dot-product attention: O = softmax(scale · Q Kᵀ [+ mask]) V.
 *
 * Lowers to a flash-attention nest: a single streaming pass over the key/value axis with
 * online softmax (running max/denominator + an acc[Dv] register tile), so the [Nq, Nk]
 * score matrix is never materialized. Operand layouts carry strides, so non-contiguous
 * Q/K/V/O (permuted [..., Nq, D] views, sliced KV caches) are addressed directly.
 *
 * Operands (leading dims [...] are broadcast batch/head axes):
 *   O [..., Nq, Dv]  in-place output   (input idx 0)
 *   Q [..., Nq, D]                     (input idx 1)
 *   K [..., Nk, D]                     (input idx 2)
 *   V [..., Nk, Dv]                    (input idx 3)
 *   M [..., Nq, Nk]  optional add mask (input idx 4)
 */
class AttentionNode : public TensorNode {
private:
    TensorLayout o_layout_;
    TensorLayout q_layout_;
    TensorLayout k_layout_;
    TensorLayout v_layout_;
    std::optional<TensorLayout> mask_layout_;
    double scale_;
    bool is_causal_;
    QuantizationType fixed_quantization_;

public:
    static auto constexpr O_INPUT_IDX = 0;
    static auto constexpr Q_INPUT_IDX = 1;
    static auto constexpr K_INPUT_IDX = 2;
    static auto constexpr V_INPUT_IDX = 3;
    static auto constexpr MASK_INPUT_IDX = 4;

    /** @brief Construct an attention node without an additive mask. */
    AttentionNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const TensorLayout& o_layout,
        const TensorLayout& q_layout,
        const TensorLayout& k_layout,
        const TensorLayout& v_layout,
        double scale,
        bool is_causal,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    /** @brief Construct an attention node with an additive mask operand. */
    AttentionNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const TensorLayout& o_layout,
        const TensorLayout& q_layout,
        const TensorLayout& k_layout,
        const TensorLayout& v_layout,
        const TensorLayout& mask_layout,
        double scale,
        bool is_causal,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    const TensorLayout& o_layout() const { return o_layout_; }
    const TensorLayout& q_layout() const { return q_layout_; }
    const TensorLayout& k_layout() const { return k_layout_; }
    const TensorLayout& v_layout() const { return v_layout_; }
    const std::optional<TensorLayout>& mask_layout() const { return mask_layout_; }

    double scale() const { return scale_; }
    bool is_causal() const { return is_causal_; }
    bool has_mask() const { return mask_layout_.has_value(); }

    /** @brief Sequence length of the query axis (second-to-last dim of Q). */
    symbolic::Expression seq_q() const { return q_layout_.get_dim_innermost(1); }
    /** @brief Sequence length of the key/value axis (second-to-last dim of K). */
    symbolic::Expression seq_k() const { return k_layout_.get_dim_innermost(1); }
    /** @brief Query/key head dimension (last dim of Q). */
    symbolic::Expression head_dim() const { return q_layout_.get_dim_innermost(0); }
    /** @brief Value head dimension (last dim of V). */
    symbolic::Expression head_dim_v() const { return v_layout_.get_dim_innermost(0); }

    QuantizationType quantization() const { return fixed_quantization_; }
    void set_quantization(QuantizationType quant) { fixed_quantization_ = quant; }

    void validate(const Function& function) const override;

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    bool supports_integer_types() const override { return false; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    symbolic::Expression flop() const override;
};

class AttentionNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
