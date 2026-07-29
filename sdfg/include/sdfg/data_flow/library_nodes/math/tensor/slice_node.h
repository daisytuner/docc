#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Slice("ml::Slice");

/** @brief Tensor slice node that extracts a strided range along a single dimension.
 *
 * Implements the semantics of `aten.slice.Tensor(self, dim, start, end, step)`. The
 * output tensor has the same shape as the input except along `dim`, where its size is
 * `ceil((end - start) / step)`.
 *
 * The `start`, `end` and `step` values are expected to be already normalized to
 * non-negative, in-bounds integers (see the PyTorch frontend `SliceParser`).
 *
 * The expansion is a map nest over the output dimensions with a single copy tasklet.
 * The source index along `dim` is `start + _i * step`; all other dimensions are copied
 * one-to-one.
 */
class SliceNode : public TensorNode {
private:
    std::vector<symbolic::Expression> input_shape_;
    long long dim_;
    long long start_;
    long long end_;
    long long step_;

public:
    SliceNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& input_shape,
        long long dim,
        long long start,
        long long end,
        long long step,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static auto constexpr RESULT_PTR_IDX = 0;
    static auto constexpr X_INPUT_IDX = 1;

    const std::vector<symbolic::Expression>& input_shape() const;
    long long dim() const;
    long long start() const;
    long long end() const;
    long long step() const;

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

class SliceNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
