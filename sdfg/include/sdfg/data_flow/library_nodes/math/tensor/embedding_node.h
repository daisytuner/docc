#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Embedding("ml::Embedding");

/** @brief Embedding lookup node implementing `aten.embedding(weight, indices)`.
 *
 * Gathers rows from a 2D `weight` tensor of shape `[num_embeddings, embedding_dim]`
 * using an integer `indices` tensor of arbitrary shape. The output shape is:
 *
 *   index_shape ++ [embedding_dim]
 *
 * so that `Y[i_0, ..., i_{k-1}, j] = W[I[i_0, ..., i_{k-1}], j]`.
 *
 * The `padding_idx`, `scale_grad_by_freq` and `sparse` arguments of
 * `aten.embedding` only affect the backward pass and are therefore irrelevant to
 * this forward-only lowering; they are intentionally not modelled here.
 *
 * The index values are expected to be already normalized to non-negative,
 * in-bounds integers.
 *
 * The expansion is a map nest over the output dimensions. In each iteration the
 * index value is first loaded from the index tensor into a scalar symbol, which is
 * then used as the data-dependent row coordinate for a single gathering copy
 * tasklet reading from `W`.
 */
class EmbeddingNode : public TensorNode {
private:
    std::vector<symbolic::Expression> weight_shape_;
    std::vector<symbolic::Expression> index_shape_;

public:
    EmbeddingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& weight_shape,
        const std::vector<symbolic::Expression>& index_shape,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static auto constexpr RESULT_PTR_IDX = 0;
    static auto constexpr W_INPUT_IDX = 1;
    static auto constexpr INDEX_IDX = 2;

    const std::vector<symbolic::Expression>& weight_shape() const;
    const std::vector<symbolic::Expression>& index_shape() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    bool supports_integer_types() const override;

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class EmbeddingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
