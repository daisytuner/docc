#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/leaky_relu_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

LeakyReLUNode::LeakyReLUNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_LeakyReLU,
          shape,
          "Y",
          {"X", "alpha"},
          quantization,
          impl_type
      ) {
}

ElementWiseDataflowTensorNode::ElementOutput LeakyReLUNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input0 = needed_inputs.at(0);
    auto& alpha_input = needed_inputs.at(1);

    types::Scalar scalar_type(input0.required_type);

    // leaky_relu(x) = max(x, 0) + alpha * min(x, 0)

    // x
    auto& assign_op = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    input0.consumer = &assign_op;
    input0.input_conn_index = 0;
    auto& output_node_x = create_tmp_access_node(builder, block, "tmp_lkyrl_x_", scalar_type);
    builder.add_computational_memlet(block, assign_op, "_out", output_node_x, {}, scalar_type);

    // max(x, 0)
    auto& zero_node = builder.add_constant(block, "0.0", scalar_type);
    auto& max_op = builder.add_library_node<
        math::cmath::CMathNode>(block, block.debug_info(), cmath::CMathFunction::fmax, input0.required_type);
    builder.add_computational_memlet(block, output_node_x, max_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, zero_node, max_op, "_in2", {}, scalar_type);
    auto& output_node_max = create_tmp_access_node(builder, block, "tmp_lkyrl_max_", scalar_type);
    builder.add_computational_memlet(block, max_op, "_out", output_node_max, {}, scalar_type);

    // min(x, 0)
    auto& min_op = builder.add_library_node<
        math::cmath::CMathNode>(block, block.debug_info(), cmath::CMathFunction::fmin, input0.required_type);
    builder.add_computational_memlet(block, output_node_x, min_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, zero_node, min_op, "_in2", {}, scalar_type);
    auto& output_node_min = create_tmp_access_node(builder, block, "tmp_lkyrl_min_", scalar_type);
    builder.add_computational_memlet(block, min_op, "_out", output_node_min, {}, scalar_type);

    // alpha * min(x, 0) + max(x, 0)
    auto& fma_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    alpha_input.consumer = &fma_op;
    alpha_input.input_conn_index = 0;
    builder.add_computational_memlet(block, output_node_min, fma_op, "_in2", {}, scalar_type);
    builder.add_computational_memlet(block, output_node_max, fma_op, "_in3", {}, scalar_type);

    return {.producer = &fma_op, .output_conn_index = 0, .type = input0.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> LeakyReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new LeakyReLUNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
