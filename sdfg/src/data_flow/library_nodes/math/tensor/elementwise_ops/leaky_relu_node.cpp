#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/leaky_relu_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

LeakyReLUNode::LeakyReLUNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_LeakyReLU, shape) {
    this->inputs_.push_back("alpha");
}

bool LeakyReLUNode::expand_operation(
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
    auto& output_node_max = builder.add_access(code_block, output_name);
    auto& output_node_mul = builder.add_access(code_block, output_name);

    types::Tensor scalar_tensor(types::Scalar(output_type.primitive_type()), {});

    // max(x, 0)
    {
        auto& zero_node = builder.add_constant(code_block, "0.0", types::Scalar(input_type.primitive_type()));
        auto& tasklet = builder.add_library_node<math::cmath::CMathNode>(
            code_block, code_block.debug_info(), cmath::CMathFunction::fmax, input_type.primitive_type()
        );
        builder.add_computational_memlet(code_block, input_node, tasklet, "_in1", subset, input_type);
        builder.add_computational_memlet(code_block, zero_node, tasklet, "_in2", {}, scalar_tensor);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_max, subset, output_type);
    }
    // alpha * x
    {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(code_block, output_node_max, tasklet, "_in2", subset, output_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_mul, subset, output_type);
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> LeakyReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new LeakyReLUNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
