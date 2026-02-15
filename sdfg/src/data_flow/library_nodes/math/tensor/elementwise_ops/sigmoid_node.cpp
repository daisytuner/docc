#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sigmoid_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

SigmoidNode::SigmoidNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Sigmoid, shape) {}

bool SigmoidNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::Tensor& input_type,
    const types::Tensor& output_type,
    const data_flow::Subset& subset
) {
    // Add code
    auto& code_block = builder.add_block(body);
    auto& input_node = builder.add_access(code_block, input_name);
    auto& output_node_neg = builder.add_access(code_block, output_name);
    auto& output_node_exp = builder.add_access(code_block, output_name);
    auto& output_node_add = builder.add_access(code_block, output_name);
    auto& output_node_div = builder.add_access(code_block, output_name);

    sdfg::types::Scalar element_type(output_type.primitive_type());
    types::Tensor scalar_tensor(types::Scalar(output_type.primitive_type()), {});

    // -x
    {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::fp_neg, "_out", {"_in"});
        builder.add_computational_memlet(code_block, input_node, tasklet, "_in", subset, input_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_neg, subset, output_type);
    }
    // exp(x)
    {
        auto& tasklet = builder.add_library_node<math::cmath::CMathNode>(
            code_block, code_block.debug_info(), cmath::CMathFunction::exp, input_type.primitive_type()
        );
        builder.add_computational_memlet(code_block, output_node_neg, tasklet, "_in1", subset, output_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_exp, subset, output_type);
    }

    // 1 + x
    {
        auto& one_node = builder.add_constant(code_block, "1.0", element_type);
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(code_block, one_node, tasklet, "_in1", {}, scalar_tensor);
        builder.add_computational_memlet(code_block, output_node_exp, tasklet, "_in2", subset, output_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_add, subset, output_type);
    }
    // 1.0 / x
    {
        auto& one_node = builder.add_constant(code_block, "1.0", element_type);
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(code_block, one_node, tasklet, "_in1", {}, scalar_tensor);
        builder.add_computational_memlet(code_block, output_node_add, tasklet, "_in2", subset, output_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_div, subset, output_type);
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> SigmoidNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new SigmoidNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
