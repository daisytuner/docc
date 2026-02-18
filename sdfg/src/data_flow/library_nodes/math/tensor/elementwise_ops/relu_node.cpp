#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

ReLUNode::ReLUNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_ReLU, shape) {}

bool ReLUNode::expand_operation(
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
    types::Scalar base_type(input_type.primitive_type());
    types::Tensor scalar_tensor(base_type.primitive_type(), {});

    auto& code_block = builder.add_block(body);
    auto& input_node_new = builder.add_access(code_block, input_name);
    auto& output_node_new = builder.add_access(code_block, output_name);
    auto& zero_node = builder.add_constant(code_block, "0.0", base_type);

    auto& tasklet = builder.add_library_node<math::cmath::CMathNode>(
        code_block, code_block.debug_info(), cmath::CMathFunction::fmax, input_type.primitive_type()
    );

    builder.add_computational_memlet(code_block, zero_node, tasklet, "_in1", {}, scalar_tensor);
    builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in2", subset, input_type);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, subset, output_type);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> ReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new ReLUNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
