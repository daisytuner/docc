#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/hard_sigmoid_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

HardSigmoidNode::HardSigmoidNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_HardSigmoid, shape) {
    this->inputs_.push_back("alpha");
    this->inputs_.push_back("beta");
}

bool HardSigmoidNode::expand_operation(
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
    auto& output_node_fma = builder.add_access(code_block, output_name);
    auto& output_node_min = builder.add_access(code_block, output_name);
    auto& output_node_max = builder.add_access(code_block, output_name);

    types::Tensor scalar_tensor(types::Scalar(output_type.primitive_type()), {});

    // alpha * x + beta
    {
        auto& tasklet =
            builder.add_tasklet(code_block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(code_block, input_node, tasklet, "_in1", subset, input_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_fma, subset, output_type);
    }
    // min(1, x)
    {
        auto& one_node = builder.add_constant(code_block, "1.0f", types::Scalar(output_type.primitive_type()));
        auto& tasklet = builder.add_library_node<math::cmath::CMathNode>(
            code_block, code_block.debug_info(), cmath::CMathFunction::fmin, output_type.primitive_type()
        );
        builder.add_computational_memlet(code_block, output_node_fma, tasklet, "_in1", subset, output_type);
        builder.add_computational_memlet(code_block, one_node, tasklet, "_in2", {}, scalar_tensor);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_min, subset, output_type);
    }
    // max(0, x)
    {
        auto& zero_node = builder.add_constant(code_block, "0.0f", types::Scalar(output_type.primitive_type()));
        auto& tasklet = builder.add_library_node<math::cmath::CMathNode>(
            code_block, code_block.debug_info(), cmath::CMathFunction::fmax, output_type.primitive_type()
        );
        builder.add_computational_memlet(code_block, output_node_min, tasklet, "_in1", subset, output_type);
        builder.add_computational_memlet(code_block, zero_node, tasklet, "_in2", {}, scalar_tensor);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_max, subset, output_type);
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> HardSigmoidNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new HardSigmoidNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
