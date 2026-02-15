#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/pow_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

PowNode::PowNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseBinaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Pow, shape) {}

bool PowNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name_a,
    const std::string& input_name_b,
    const std::string& output_name,
    const types::Tensor& input_type_a,
    const types::Tensor& input_type_b,
    const types::Tensor& output_type,
    const data_flow::Subset& subset
) {
    auto& code_block = builder.add_block(body);
    data_flow::AccessNode* input_node_a;
    if (builder.subject().exists(input_name_a)) {
        input_node_a = &builder.add_access(code_block, input_name_a);
    } else {
        types::Scalar const_type(input_type_a.primitive_type());
        input_node_a = &builder.add_constant(code_block, input_name_a, const_type);
    }
    data_flow::AccessNode* input_node_b;
    if (builder.subject().exists(input_name_b)) {
        input_node_b = &builder.add_access(code_block, input_name_b);
    } else {
        types::Scalar const_type(input_type_b.primitive_type());
        input_node_b = &builder.add_constant(code_block, input_name_b, const_type);
    }
    auto& output_node = builder.add_access(code_block, output_name);

    auto& tasklet = builder.add_library_node<math::cmath::CMathNode>(
        code_block, code_block.debug_info(), cmath::CMathFunction::pow, input_type_a.primitive_type()
    );

    builder.add_computational_memlet(code_block, *input_node_a, tasklet, "_in1", subset, input_type_a);
    builder.add_computational_memlet(code_block, *input_node_b, tasklet, "_in2", subset, input_type_b);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node, subset, output_type);

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> PowNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new PowNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
