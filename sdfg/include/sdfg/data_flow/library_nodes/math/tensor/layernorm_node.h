#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg::math::tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_LayerNorm("ml::LayerNorm");

/**
 * Applies layer normalization over the trailing `num_normalized_dims` dimensions of the input.
 * Mean and variance are computed on the fly for each "row" (the leading dimensions).
 * The optional affine scale (Gamma) and bias (Beta) are shaped like the normalized dimensions.
 */
class LayerNormNode : public TensorNode {
    /**
     * Layout of input and normalized output
     */
    TensorLayout layout_;
    QuantizationType quantization_;
    /**
     * Number of trailing dimensions that are normalized over (the rank of normalized_shape).
     */
    size_t num_normalized_dims_;
    /**
     * Whether a learnable scale (Gamma) is applied.
     */
    bool affine_;
    /**
     * Whether a learnable bias (Beta) is applied. Only meaningful together with affine_.
     */
    bool has_bias_;

public:
    LayerNormNode(
        size_t element_id,
        const DebugInfo& debug_info,
        graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        TensorLayout layout,
        QuantizationType quantization,
        size_t num_normalized_dims,
        bool affine,
        bool has_bias,
        data_flow::ImplementationType impl_type = data_flow::ImplementationType_NONE
    );

    const TensorLayout& layernorm_layout() const { return layout_; }

    /**
     * Number of trailing dimensions that are normalized over.
     */
    size_t num_normalized_dims() const { return num_normalized_dims_; }

    bool affine() const { return affine_; }

    bool has_bias() const { return has_bias_; }

    QuantizationType quantization() const;

    void set_quantization(const QuantizationType quant);

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    symbolic::Expression flop() const override;

    bool supports_integer_types() const override { return false; }

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class LayerNormNodeSerializer : public serializer::LibraryNodeSerializer {
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace sdfg::math::tensor
