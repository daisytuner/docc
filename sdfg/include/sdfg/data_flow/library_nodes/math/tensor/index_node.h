#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Index("ml::Index");

/** @brief Tensor advanced-indexing node implementing `aten.index.Tensor(self, indices)`.
 *
 * Gathers elements from the input tensor `X` using one or more integer index tensors.
 * This models the subset of PyTorch advanced indexing where the index tensors occupy a
 * contiguous block of the input dimensions (i.e. `indices = [None, ..., I0, I1, ..., None, ...]`
 * with the non-`None` entries adjacent) and all broadcast to a common shape.
 *
 * `dim_offset` is the position of the first indexed dimension in `X`, `num_indices` is the
 * number of index tensors and `index_shape` is their common (broadcast) shape. The output
 * shape is:
 *
 *   input_shape[0 : dim_offset] ++ index_shape ++ input_shape[dim_offset + num_indices :]
 *
 * The index values are expected to be already normalized to non-negative, in-bounds integers.
 *
 * The expansion is a map nest over the output dimensions. In each iteration the index values
 * are first loaded from the index tensors into scalar symbols, which are then used as the
 * data-dependent coordinates for a single gathering copy tasklet reading from `X`.
 */
class IndexNode : public TensorNode {
private:
    std::vector<symbolic::Expression> input_shape_;
    std::vector<symbolic::Expression> index_shape_;
    long long dim_offset_;
    long long num_indices_;

public:
    IndexNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& input_shape,
        const std::vector<symbolic::Expression>& index_shape,
        long long dim_offset,
        long long num_indices,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static auto constexpr RESULT_PTR_IDX = 0;
    static auto constexpr X_INPUT_IDX = 1;
    static auto constexpr FIRST_INDEX_IDX = 2;

    const std::vector<symbolic::Expression>& input_shape() const;
    const std::vector<symbolic::Expression>& index_shape() const;
    long long dim_offset() const;
    long long num_indices() const;

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

class IndexNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
