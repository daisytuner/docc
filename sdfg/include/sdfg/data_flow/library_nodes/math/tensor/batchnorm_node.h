#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg::math::tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_BatchNorm("ml::BatchNorm");

/**
 * In N, C, D0...Dn tensor, applies normalization to per channel in C across all the inner dimensions
 */
class BatchNormNode : public TensorNode {
    /**
     * Layout of input and normalized output
     */
    TensorLayout layout_;
    QuantizationType quantization_;

public:
    BatchNormNode(
        size_t element_id,
        const DebugInfo& debug_info,
        graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        TensorLayout layout,
        QuantizationType quantization,
        data_flow::ImplementationType impl_type = data_flow::ImplementationType_NONE
    );

    const TensorLayout& batch_layout() const { return layout_; }

    /**
     * In N,C,D0...Dn layout, always C
     */
    symbolic::Expression num_features() const { return layout_.shape().at(1); }

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

class BatchNormNodeSerializer : public serializer::LibraryNodeSerializer {
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace sdfg::math::tensor
