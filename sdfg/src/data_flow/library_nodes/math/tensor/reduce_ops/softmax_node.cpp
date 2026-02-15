#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/softmax_node.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/broadcast_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/div_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/exp_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sub_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/max_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/sum_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace tensor {

SoftmaxNode::SoftmaxNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : ReduceNode(element_id, debug_info, vertex, parent, LibraryNodeType_Softmax, shape, axes, keepdims) {
    if (keepdims) {
        throw InvalidSDFGException("Unsupported attribute on library node: softmax");
    }
}

void SoftmaxNode::validate(const Function& function) const {}

bool SoftmaxNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    if (dataflow.in_degree(*this) != 1 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);

    auto& in_edge = *dataflow.in_edges(*this).begin();
    auto& out_edge = *dataflow.out_edges(*this).begin();
    auto& in_node = static_cast<data_flow::AccessNode&>(in_edge.src());
    auto& out_node = static_cast<data_flow::AccessNode&>(out_edge.dst());

    // Calculate reduced shape (for Max and Sum)
    std::vector<symbolic::Expression> reduced_shape;
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
            reduced_shape.push_back(symbolic::one());
        } else {
            reduced_shape.push_back(shape_[i]);
        }
    }

    types::Scalar element_type(this->primitive_type(dataflow));
    types::Pointer pointer_type(element_type);

    // Temporary buffers
    std::string tmp_max_name = builder.find_new_name("_softmax_max");
    builder.add_container(tmp_max_name, pointer_type);

    std::string tmp_max_bcast_name = builder.find_new_name("_softmax_max_bcast");
    builder.add_container(tmp_max_bcast_name, pointer_type);

    std::string tmp_sub_name = builder.find_new_name("_softmax_sub");
    builder.add_container(tmp_sub_name, pointer_type);

    std::string tmp_exp_name = builder.find_new_name("_softmax_exp");
    builder.add_container(tmp_exp_name, pointer_type);

    std::string tmp_sum_name = builder.find_new_name("_softmax_sum");
    builder.add_container(tmp_sum_name, pointer_type);

    std::string tmp_sum_bcast_name = builder.find_new_name("_softmax_sum_bcast");
    builder.add_container(tmp_sum_bcast_name, pointer_type);

    // Mallocs
    {
        symbolic::Expression bytes_elem = types::get_type_size(element_type, false);

        symbolic::Expression bytes_full = bytes_elem;
        for (auto& dim : this->shape_) {
            bytes_full = symbolic::mul(dim, bytes_full);
        }

        symbolic::Expression bytes_reduced = bytes_elem;
        for (auto& dim : reduced_shape) {
            bytes_reduced = symbolic::mul(dim, bytes_reduced);
        }

        auto& alloc_block = builder.add_block_before(parent, block, {}, this->debug_info());

        auto malloc_helper = [&](const std::string& name, const symbolic::Expression& size) {
            auto& access = builder.add_access(alloc_block, name);
            auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(alloc_block, this->debug_info(), size);
            builder
                .add_computational_memlet(alloc_block, malloc_node, "_ret", access, {}, pointer_type, this->debug_info());
        };

        malloc_helper(tmp_max_name, bytes_reduced);
        malloc_helper(tmp_max_bcast_name, bytes_full);
        malloc_helper(tmp_sub_name, bytes_full);
        malloc_helper(tmp_exp_name, bytes_full);
        malloc_helper(tmp_sum_name, bytes_reduced);
        malloc_helper(tmp_sum_bcast_name, bytes_full);
    }

    // 1. Max(X) -> TmpMax
    {
        auto& max_block = builder.add_block_before(parent, block, {}, this->debug_info());
        auto& max_node =
            builder.add_library_node<MaxNode>(max_block, this->debug_info(), this->shape_, this->axes_, true);

        auto& in_access = builder.add_access(max_block, in_node.data());
        auto& out_access = builder.add_access(max_block, tmp_max_name);

        builder
            .add_computational_memlet(max_block, in_access, max_node, "X", {}, in_edge.base_type(), this->debug_info());
        builder
            .add_computational_memlet(max_block, max_node, "Y", out_access, {}, out_edge.base_type(), this->debug_info());
    }

    // 1.5 Broadcast Max -> TmpMaxBcast
    {
        auto& bcast_block = builder.add_block_before(parent, block, {}, this->debug_info());
        auto& bcast_node =
            builder.add_library_node<BroadcastNode>(bcast_block, this->debug_info(), reduced_shape, this->shape_);

        auto& in_access = builder.add_access(bcast_block, tmp_max_name);
        auto& out_access = builder.add_access(bcast_block, tmp_max_bcast_name);

        builder.add_computational_memlet(
            bcast_block, in_access, bcast_node, "X", {}, out_edge.base_type(), this->debug_info()
        );
        builder.add_computational_memlet(
            bcast_block, bcast_node, "Y", out_access, {}, out_edge.base_type(), this->debug_info()
        );
    }

    // 2. Sub(X, TmpMaxBcast) -> TmpSub
    {
        auto& sub_block = builder.add_block_before(parent, block, {}, this->debug_info());
        auto& sub_node = builder.add_library_node<SubNode>(sub_block, this->debug_info(), this->shape_);

        auto& in1_access = builder.add_access(sub_block, in_node.data());
        auto& in2_access = builder.add_access(sub_block, tmp_max_bcast_name);
        auto& out_access = builder.add_access(sub_block, tmp_sub_name);

        builder
            .add_computational_memlet(sub_block, in1_access, sub_node, "A", {}, in_edge.base_type(), this->debug_info());
        builder
            .add_computational_memlet(sub_block, in2_access, sub_node, "B", {}, out_edge.base_type(), this->debug_info());
        builder
            .add_computational_memlet(sub_block, sub_node, "C", out_access, {}, out_edge.base_type(), this->debug_info());
    }

    // 3. Exp(TmpSub) -> TmpExp
    {
        auto& exp_block = builder.add_block_before(parent, block, {}, this->debug_info());
        auto& exp_node = builder.add_library_node<ExpNode>(exp_block, this->debug_info(), this->shape_);

        auto& in_access = builder.add_access(exp_block, tmp_sub_name);
        auto& out_access = builder.add_access(exp_block, tmp_exp_name);

        builder
            .add_computational_memlet(exp_block, in_access, exp_node, "X", {}, out_edge.base_type(), this->debug_info());
        builder
            .add_computational_memlet(exp_block, exp_node, "Y", out_access, {}, out_edge.base_type(), this->debug_info());
    }

    // 4. Sum(TmpExp) -> TmpSum
    {
        auto& sum_block = builder.add_block_before(parent, block, {}, this->debug_info());
        auto& sum_node =
            builder.add_library_node<SumNode>(sum_block, this->debug_info(), this->shape_, this->axes_, true);

        auto& in_access = builder.add_access(sum_block, tmp_exp_name);
        auto& out_access = builder.add_access(sum_block, tmp_sum_name);

        builder
            .add_computational_memlet(sum_block, in_access, sum_node, "X", {}, out_edge.base_type(), this->debug_info());
        builder
            .add_computational_memlet(sum_block, sum_node, "Y", out_access, {}, out_edge.base_type(), this->debug_info());
    }

    // 4.5 Broadcast Sum -> TmpSumBcast
    {
        auto& bcast_block = builder.add_block_before(parent, block, {}, this->debug_info());
        auto& bcast_node =
            builder.add_library_node<BroadcastNode>(bcast_block, this->debug_info(), reduced_shape, this->shape_);

        auto& in_access = builder.add_access(bcast_block, tmp_sum_name);
        auto& out_access = builder.add_access(bcast_block, tmp_sum_bcast_name);

        builder.add_computational_memlet(
            bcast_block, in_access, bcast_node, "X", {}, out_edge.base_type(), this->debug_info()
        );
        builder.add_computational_memlet(
            bcast_block, bcast_node, "Y", out_access, {}, out_edge.base_type(), this->debug_info()
        );
    }

    // 5. Div(TmpExp, TmpSumBcast) -> Output
    {
        auto& div_block = builder.add_block_before(parent, block, {}, this->debug_info());
        auto& div_node = builder.add_library_node<DivNode>(div_block, this->debug_info(), this->shape_);

        auto& in1_access = builder.add_access(div_block, tmp_exp_name);
        auto& in2_access = builder.add_access(div_block, tmp_sum_bcast_name);
        auto& out_access = builder.add_access(div_block, out_node.data());

        builder
            .add_computational_memlet(div_block, in1_access, div_node, "A", {}, out_edge.base_type(), this->debug_info());
        builder
            .add_computational_memlet(div_block, in2_access, div_node, "B", {}, out_edge.base_type(), this->debug_info());
        builder
            .add_computational_memlet(div_block, div_node, "C", out_access, {}, out_edge.base_type(), this->debug_info());
    }

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

std::unique_ptr<data_flow::DataFlowNode> SoftmaxNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new SoftmaxNode(element_id, this->debug_info(), vertex, parent, this->shape_, this->axes_)
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg
