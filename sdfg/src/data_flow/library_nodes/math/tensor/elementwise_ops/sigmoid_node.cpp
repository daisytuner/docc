#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sigmoid_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

SigmoidNode::SigmoidNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Sigmoid, shape, "Y", {"X"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput SigmoidNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input = needed_inputs.at(0);

    sdfg::types::Scalar scalar_type(input.required_type);

    // -x
    auto& first_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_neg, "_out", {"_in"});
    input.consumer = &first_op;
    input.input_conn_index = 0;
    auto& output_node_neg = create_tmp_access_node(builder, block, "tmp_sgmd_ng_", scalar_type);
    builder.add_computational_memlet(block, first_op, "_out", output_node_neg, {}, scalar_type);
    // exp(x)
    auto& exp_op = builder.add_library_node<
        math::cmath::CMathNode>(block, block.debug_info(), cmath::CMathFunction::exp, input.required_type);
    builder.add_computational_memlet(block, output_node_neg, exp_op, "_in1", {}, scalar_type);
    auto& output_node_exp = create_tmp_access_node(builder, block, "tmp_sgmd_exp_", scalar_type);
    builder.add_computational_memlet(block, exp_op, "_out", output_node_exp, {}, scalar_type);

    // 1 + x
    auto& one_node = builder.add_constant(block, "1.0", scalar_type);
    auto& add_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, one_node, add_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, output_node_exp, add_op, "_in2", {}, scalar_type);
    auto& output_node_add = create_tmp_access_node(builder, block, "tmp_sgmd_add_", scalar_type);
    builder.add_computational_memlet(block, add_op, "_out", output_node_add, {}, scalar_type);
    // 1.0 / x
    auto& last_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, one_node, last_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, output_node_add, last_op, "_in2", {}, scalar_type);
    return {.producer = &last_op, .output_conn_index = 0, .type = input.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> SigmoidNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new SigmoidNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
