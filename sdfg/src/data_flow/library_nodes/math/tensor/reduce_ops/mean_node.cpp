#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/mean_node.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/div_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/sum_node.h"

namespace sdfg {
namespace math {
namespace tensor {

MeanNode::MeanNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : ReduceNode(element_id, debug_info, vertex, parent, LibraryNodeType_Mean, shape, axes, keepdims) {}

passes::LibNodeExpander::ExpandOutcome MeanNode::expand_inner(
    passes::LibNodeExpander::AccessNodeExpand& expansion,
    structured_control_flow::Block& block,
    const data_flow::Memlet* iedge_input,
    const data_flow::Memlet* iedge_result,
    const std::vector<symbolic::Expression>& output_shape,
    const std::vector<int64_t>& sorted_axes
) {
    auto& out_type = iedge_result->base_type();
    auto& in_type = iedge_input->base_type();

    auto& seq = expansion.replace_with_sequence();
    auto& builder = expansion.builder();

    // Create SumNode
    auto& sum_block = builder.add_block(seq, {}, this->debug_info());
    auto& sum_node = builder.add_library_node<SumNode>(sum_block, this->debug_info(), shape_, axes_, keepdims_);

    // Create intermediate buffer for Sum result
    auto& sum_in_node = expansion.add_scalar_input_access(sum_block, X_INPUT_IDX);
    auto& sum_out_node = expansion.add_scalar_input_access(sum_block, RESULT_PTR_IDX);

    // Connect Input -> Sum -> Tmp
    builder.add_computational_memlet(sum_block, sum_in_node, sum_node, "X", {}, in_type, this->debug_info());
    builder.add_computational_memlet(sum_block, sum_out_node, sum_node, "Y", {}, out_type, this->debug_info());

    // Create Count (symbolically)
    auto count_container = builder.find_new_name("_mean_count");
    builder.add_container(count_container, types::Scalar(types::PrimitiveType::Int64));

    symbolic::Expression count_expr = symbolic::one();
    for (auto axis : axes_) {
        int64_t ax = axis;
        if (ax < 0) ax += shape_.size();
        symbolic::Expression dim = shape_[ax];
        count_expr = symbolic::mul(count_expr, dim);
    }
    auto& count_block = builder.add_block(seq, {{symbolic::symbol(count_container), count_expr}}, this->debug_info());

    // Create DivNode
    auto& div_block = builder.add_block(seq, {}, this->debug_info());
    auto& div_node = builder.add_library_node<DivNode>(div_block, this->debug_info(), output_shape);

    // Connect Tmp -> Div (A)
    auto& div_in_node = expansion.add_scalar_input_access(div_block, RESULT_PTR_IDX);
    builder.add_computational_memlet(div_block, div_in_node, div_node, "A", {}, out_type, this->debug_info());

    // Connect Count -> Div (B)
    types::Tensor scalar_tensor(out_type.primitive_type(), {});
    auto& div_count_node = builder.add_access(div_block, count_container, this->debug_info());
    builder.add_computational_memlet(div_block, div_count_node, div_node, "B", {}, scalar_tensor, this->debug_info());

    // Connect Div -> Output (C)
    builder.add_computational_memlet(div_block, div_in_node, div_node, "C", {}, out_type, this->debug_info());

    return expansion.successfully_expanded();
}

bool MeanNode::expand_reduction(
    passes::LibNodeExpander::AccessNodeExpand& expansion,
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& body,
    const types::Tensor& input_type,
    const types::Tensor& output_type,
    const data_flow::Subset& input_subset,
    const data_flow::Subset& output_subset
) {
    throw std::runtime_error("MeanNode::expand_reduction should not be called");
}

std::string MeanNode::identity(types::PrimitiveType primitive_type) const { return "0"; }

std::unique_ptr<data_flow::DataFlowNode> MeanNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new MeanNode(element_id, debug_info_, vertex, parent, shape_, axes_, keepdims_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
