#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/mean_node.h"

#include "sdfg/analysis/scope_analysis.h"
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

bool MeanNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    auto& in_edge = *dataflow.in_edges(*this).begin();
    auto& out_edge = *dataflow.out_edges(*this).begin();
    auto& in_node = static_cast<data_flow::AccessNode&>(in_edge.src());
    auto& out_node = static_cast<data_flow::AccessNode&>(out_edge.dst());

    // Calculate output shape
    std::vector<symbolic::Expression> output_shape;
    std::vector<int64_t> sorted_axes = axes_;
    // Normalize negative axes
    for (auto& axis : sorted_axes) {
        if (axis < 0) {
            axis = static_cast<int64_t>(shape_.size()) + axis;
        }
        // Validate axis is in bounds
        if (axis < 0 || axis >= static_cast<int64_t>(shape_.size())) {
            throw InvalidSDFGException(
                "Library Node: Axis value out of bounds. Axis: " + std::to_string(axis) +
                " Shape size: " + std::to_string(shape_.size())
            );
        }
    }
    std::sort(sorted_axes.begin(), sorted_axes.end());

    for (size_t i = 0; i < shape_.size(); ++i) {
        bool is_axis = false;
        for (auto axis : sorted_axes) {
            if (axis == (int64_t) i) {
                is_axis = true;
                break;
            }
        }

        if (is_axis) {
            if (keepdims_) {
                output_shape.push_back(symbolic::one());
            }
        } else {
            output_shape.push_back(shape_[i]);
        }
    }

    // Create SumNode
    auto& sum_block = builder.add_block_before(parent, block, {}, this->debug_info());
    auto& sum_node = builder.add_library_node<SumNode>(sum_block, this->debug_info(), shape_, axes_, keepdims_);

    // Create intermediate buffer for Sum result
    auto& sum_in_node = builder.add_access(sum_block, in_node.data(), this->debug_info());
    auto& sum_out_node = builder.add_access(sum_block, out_node.data(), this->debug_info());

    // Connect Input -> Sum -> Tmp
    builder.add_computational_memlet(sum_block, sum_in_node, sum_node, "X", {}, in_edge.base_type(), this->debug_info());
    builder
        .add_computational_memlet(sum_block, sum_node, "Y", sum_out_node, {}, out_edge.base_type(), this->debug_info());

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
    auto& count_block =
        builder.add_block_before(parent, block, {{symbolic::symbol(count_container), count_expr}}, this->debug_info());

    // Create DivNode
    auto& div_block = builder.add_block_before(parent, block, transition.assignments(), this->debug_info());
    auto& div_node = builder.add_library_node<DivNode>(div_block, this->debug_info(), output_shape);

    // Connect Tmp -> Div (A)
    auto& div_in_node = builder.add_access(div_block, out_node.data(), this->debug_info());
    builder
        .add_computational_memlet(div_block, div_in_node, div_node, "A", {}, out_edge.base_type(), this->debug_info());

    // Connect Count -> Div (B)
    types::Tensor scalar_tensor(out_edge.base_type().primitive_type(), {});
    auto& div_count_node = builder.add_access(div_block, count_container, this->debug_info());
    builder.add_computational_memlet(div_block, div_count_node, div_node, "B", {}, scalar_tensor, this->debug_info());

    // Connect Div -> Output (C)
    auto& div_out_node = builder.add_access(div_block, out_node.data(), this->debug_info());
    builder
        .add_computational_memlet(div_block, div_node, "C", div_out_node, {}, out_edge.base_type(), this->debug_info());


    // Cleanup
    builder.remove_memlet(block, in_edge);
    builder.remove_memlet(block, out_edge);
    builder.remove_node(block, in_node);
    builder.remove_node(block, out_node);
    builder.remove_node(block, *this);

    int last_index = parent.index(block);
    builder.remove_child(parent, last_index);

    return true;
}

bool MeanNode::expand_reduction(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
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
