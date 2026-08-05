#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_TensorConcat("ml::Concat");

/** @brief Tensor concatenation node that can concatenate a number of tensors into another tensor
 *
 * The tensors can be multi-dimensional and the concatenation dimension is given (dim). All shapes
 * of all tensors must be the same except for the conecatenation dimension. For this dimension, the
 * dimensions of all input tensors must add up to the output tensor.
 *
 * The expansion is done with a map nest over the output dimensions. Inside, a if/else structure
 * chooses the right tensor to copy from.
 */
class ConcatNode : public TensorNode {
private:
    TensorLayout result_layout_;
    std::vector<TensorLayout> tensor_layouts_;
    long long dim_;

public:
    ConcatNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::string& result,
        const TensorLayout& result_layout,
        const std::vector<std::string>& tensors,
        const std::vector<TensorLayout>& tensor_layouts,
        long long dim,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    const std::string& result() const;
    const TensorLayout& result_layout() const;

    std::vector<std::string> tensors() const;
    const std::vector<TensorLayout>& tensor_layouts() const;

    long long dim() const;

    void validate(const Function& function) const override;

    virtual bool supports_integer_types() const override;

    virtual passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    virtual std::string toStr() const override;

    virtual symbolic::SymbolSet symbols() const override;

    virtual symbolic::Expression flop() const override;

    virtual std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

class ConcatNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
