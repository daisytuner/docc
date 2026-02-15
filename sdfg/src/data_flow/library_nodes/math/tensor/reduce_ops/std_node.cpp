#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/std_node.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/mul_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sqrt_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sub_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/mean_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace tensor {

StdNode::StdNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : ReduceNode(element_id, debug_info, vertex, parent, LibraryNodeType_Std, shape, axes, keepdims) {}

bool StdNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
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

    types::Scalar element_type(this->primitive_type(dataflow));
    types::Pointer pointer_type(element_type);

    std::string tmp_x2_name = builder.find_new_name("_std_x2");
    builder.add_container(tmp_x2_name, pointer_type);
    std::string tmp_mean_x2_name = builder.find_new_name("_std_mean_x2");
    std::string tmp_mean_x_name = builder.find_new_name("_std_mean_x");

    symbolic::Expression bytes_in = types::get_type_size(element_type, false);
    for (auto& dim : this->shape_) {
        bytes_in = symbolic::mul(dim, bytes_in);
    }
    {
        auto& alloc_block = builder.add_block_before(parent, block, {}, this->debug_info());
        auto& tmp_x2_name_access = builder.add_access(alloc_block, tmp_x2_name);
        auto& tmp_x2_name_malloc_node =
            builder.add_library_node<stdlib::MallocNode>(alloc_block, this->debug_info(), bytes_in);
        builder.add_computational_memlet(
            alloc_block, tmp_x2_name_malloc_node, "_ret", tmp_x2_name_access, {}, pointer_type, this->debug_info()
        );
    }

    if (!output_shape.empty()) {
        symbolic::Expression bytes_out = types::get_type_size(element_type, false);
        for (auto& dim : output_shape) {
            bytes_out = symbolic::mul(dim, bytes_out);
        }
        builder.add_container(tmp_mean_x2_name, pointer_type);
        {
            auto& alloc_block = builder.add_block_before(parent, block, {}, this->debug_info());
            auto& tmp_mean_x2_name_access = builder.add_access(alloc_block, tmp_mean_x2_name);
            auto& tmp_mean_x2_name_malloc_node =
                builder.add_library_node<stdlib::MallocNode>(alloc_block, this->debug_info(), bytes_out);
            builder.add_computational_memlet(
                alloc_block,
                tmp_mean_x2_name_malloc_node,
                "_ret",
                tmp_mean_x2_name_access,
                {},
                pointer_type,
                this->debug_info()
            );
        }

        builder.add_container(tmp_mean_x_name, pointer_type);
        {
            auto& alloc_block = builder.add_block_before(parent, block, {}, this->debug_info());
            auto& tmp_mean_x_name_access = builder.add_access(alloc_block, tmp_mean_x_name);
            auto& tmp_mean_x_name_malloc_node =
                builder.add_library_node<stdlib::MallocNode>(alloc_block, this->debug_info(), bytes_out);
            builder.add_computational_memlet(
                alloc_block,
                tmp_mean_x_name_malloc_node,
                "_ret",
                tmp_mean_x_name_access,
                {},
                pointer_type,
                this->debug_info()
            );
        }
    } else {
        builder.add_container(tmp_mean_x2_name, element_type);
        builder.add_container(tmp_mean_x_name, element_type);
    }

    // 1. X^2
    auto& pow_block = builder.add_block_before(parent, block, {}, this->debug_info());
    auto& pow_in_node = builder.add_access(pow_block, in_node.data(), this->debug_info());
    auto& pow_out_node = builder.add_access(pow_block, tmp_x2_name, this->debug_info());

    auto& pow_node_1 = builder.add_library_node<MulNode>(pow_block, this->debug_info(), shape_);
    builder
        .add_computational_memlet(pow_block, pow_in_node, pow_node_1, "A", {}, in_edge.base_type(), this->debug_info());
    builder
        .add_computational_memlet(pow_block, pow_in_node, pow_node_1, "B", {}, in_edge.base_type(), this->debug_info());
    builder
        .add_computational_memlet(pow_block, pow_node_1, "C", pow_out_node, {}, in_edge.base_type(), this->debug_info());

    // 2. Mean(X^2)
    auto& mean_x2_block = builder.add_block_before(parent, block, {}, this->debug_info());
    auto& mean_x2_in_node = builder.add_access(mean_x2_block, tmp_x2_name, this->debug_info());
    auto& mean_x2_out_node = builder.add_access(mean_x2_block, tmp_mean_x2_name, this->debug_info());

    auto& mean_node_1 = builder.add_library_node<MeanNode>(mean_x2_block, this->debug_info(), shape_, axes_, keepdims_);
    builder.add_computational_memlet(
        mean_x2_block, mean_x2_in_node, mean_node_1, "X", {}, in_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        mean_x2_block, mean_node_1, "Y", mean_x2_out_node, {}, out_edge.base_type(), this->debug_info()
    );

    // 3. Mean(X)
    auto& mean_x_block = builder.add_block_before(parent, block, {}, this->debug_info());
    auto& mean_x_in_node = builder.add_access(mean_x_block, in_node.data(), this->debug_info());
    auto& mean_x_out_node = builder.add_access(mean_x_block, tmp_mean_x_name, this->debug_info());

    auto& mean_node_2 = builder.add_library_node<MeanNode>(mean_x_block, this->debug_info(), shape_, axes_, keepdims_);
    builder.add_computational_memlet(
        mean_x_block, mean_x_in_node, mean_node_2, "X", {}, in_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        mean_x_block, mean_node_2, "Y", mean_x_out_node, {}, out_edge.base_type(), this->debug_info()
    );

    // 4. Mean(X)^2
    auto& pow_mean_x_block = builder.add_block_before(parent, block, {}, this->debug_info());
    auto& pow_mean_x_in_node = builder.add_access(pow_mean_x_block, tmp_mean_x_name, this->debug_info());
    auto& pow_mean_x_out_node = builder.add_access(pow_mean_x_block, tmp_mean_x_name, this->debug_info());

    auto& pow_node_2 = builder.add_library_node<MulNode>(pow_mean_x_block, this->debug_info(), output_shape);

    builder.add_computational_memlet(
        pow_mean_x_block, pow_mean_x_in_node, pow_node_2, "A", {}, out_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        pow_mean_x_block, pow_mean_x_in_node, pow_node_2, "B", {}, out_edge.base_type(), this->debug_info()
    );
    builder.add_computational_memlet(
        pow_mean_x_block, pow_node_2, "C", pow_mean_x_out_node, {}, out_edge.base_type(), this->debug_info()
    );

    // 5. Mean(X^2) - Mean(X)^2
    auto& sub_block = builder.add_block_before(parent, block, {}, this->debug_info());
    auto& sub_in1_node = builder.add_access(sub_block, tmp_mean_x2_name, this->debug_info());
    auto& sub_in2_node = builder.add_access(sub_block, tmp_mean_x_name, this->debug_info());
    auto& sub_out_node = builder.add_access(sub_block, out_node.data(), this->debug_info());

    auto& sub_node = builder.add_library_node<SubNode>(sub_block, this->debug_info(), output_shape);
    builder
        .add_computational_memlet(sub_block, sub_in1_node, sub_node, "A", {}, out_edge.base_type(), this->debug_info());
    builder
        .add_computational_memlet(sub_block, sub_in2_node, sub_node, "B", {}, out_edge.base_type(), this->debug_info());
    builder
        .add_computational_memlet(sub_block, sub_node, "C", sub_out_node, {}, out_edge.base_type(), this->debug_info());

    // 6. Sqrt(...)
    auto& sqrt_block = builder.add_block_before(parent, block, transition.assignments(), this->debug_info());
    auto& sqrt_in_node = builder.add_access(sqrt_block, out_node.data(), this->debug_info());
    auto& sqrt_out_node = builder.add_access(sqrt_block, out_node.data(), this->debug_info());

    auto& sqrt_node = builder.add_library_node<SqrtNode>(sqrt_block, this->debug_info(), output_shape);
    builder
        .add_computational_memlet(sqrt_block, sqrt_in_node, sqrt_node, "X", {}, out_edge.base_type(), this->debug_info());
    builder
        .add_computational_memlet(sqrt_block, sqrt_node, "Y", sqrt_out_node, {}, out_edge.base_type(), this->debug_info());

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

bool StdNode::expand_reduction(
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
    throw std::runtime_error("StdNode::expand_reduction should not be called");
}

std::string StdNode::identity(types::PrimitiveType primitive_type) const { return "0"; }

std::unique_ptr<data_flow::DataFlowNode> StdNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new StdNode(element_id, debug_info_, vertex, parent, shape_, axes_, keepdims_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
