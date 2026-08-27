#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_EmbeddingRenorm("ml::EmbeddingRenorm");

/** @brief In-place embedding renormalization implementing `aten.embedding_renorm_`.
 *
 * Renormalizes the rows of a 2D `weight` tensor of shape `[num_embeddings, embedding_dim]` that are selected by an
 * integer `indices` tensor of arbitrary shape. For each selected row `W[idx]` whose `norm_type`-norm exceeds
 * `max_norm`, the row is scaled so that its norm equals `max_norm`:
 *
 *   norm  = ||W[idx]||_{norm_type}
 *   scale = min(1, max_norm / (norm + 1e-7))
 *   Y[idx] *= scale
 *
 * The clamped `scale` is branchless: rows already within `max_norm` are multiplied
 * by 1 and left unchanged, matching the semantics of `aten.embedding_renorm_` up to
 * the strict-inequality boundary (a measure-zero floating-point edge case).
 *
 * PyTorch deduplicates the index list before renormalizing. Because scaling a row to
 * `max_norm` makes its subsequent norm no longer exceed `max_norm`, processing the
 * (possibly duplicated) indices SEQUENTIALLY is idempotent and yields the same result
 * as deduplication; the index loops are therefore emitted as sequential maps.
 */
class EmbeddingRenormNode : public TensorNode {
private:
    TensorLayout y_layout_;
    TensorLayout weight_layout_;
    TensorLayout indices_layout_;

public:
    EmbeddingRenormNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const TensorLayout& y_layout,
        const TensorLayout& weight_layout,
        const TensorLayout& indices_layout,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static int constexpr Y_INPUT_IDX = 0;
    static int constexpr WEIGHT_INPUT_IDX = 1;
    static int constexpr INDICES_INPUT_IDX = 2;
    static int constexpr MAX_NORM_INPUT_IDX = 3;
    static int constexpr NORM_TYPE_INPUT_IDX = 4;

    const TensorLayout& y_layout() const;
    const TensorLayout& weight_layout() const;
    const TensorLayout& indices_layout() const;

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

class EmbeddingRenormNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
