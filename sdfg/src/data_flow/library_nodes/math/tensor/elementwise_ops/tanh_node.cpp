#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/tanh_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

TanhNode::TanhNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Tanh, shape) {}

bool TanhNode::expand_operation(
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
    auto& output_node = builder.add_access(code_block, output_name);

    auto& tasklet = builder.add_library_node<math::cmath::CMathNode>(
        code_block, code_block.debug_info(), cmath::CMathFunction::tanh, input_type.primitive_type()
    );

    builder.add_computational_memlet(code_block, input_node, tasklet, "_in1", subset, input_type);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node, subset, output_type);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> TanhNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new TanhNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
