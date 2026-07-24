#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/hard_sigmoid_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

HardSigmoidNode::HardSigmoidNode(
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
          LibraryNodeType_HardSigmoid,
          shape,
          "X",
          {"Y", "alpha", "beta"},
          quantization,
          impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput HardSigmoidNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input0 = needed_inputs.at(0);
    auto& input_alpha = needed_inputs.at(1);
    auto& input_beta = needed_inputs.at(2);

    types::Scalar scalar_type(input0.required_type);

    throw std::runtime_error("Hardsigmoid: untested expand");

    // alpha * x + beta
    auto& first_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    input_alpha.consumer = &first_op;
    input_alpha.input_conn_index = 0;
    input0.consumer = &first_op;
    input0.input_conn_index = 1;
    input_beta.consumer = &first_op;
    input_beta.input_conn_index = 2;
    auto& output_node_fma = create_tmp_access_node(builder, block, "tmp_hs_fma_", scalar_type);
    builder.add_computational_memlet(block, first_op, "_out", output_node_fma, {}, scalar_type);
    // min(1, x)
    auto& one_node = builder.add_constant(block, "1.0f", scalar_type);
    auto& min_op = builder.add_library_node<
        math::cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::fmin, input0.required_type);
    builder.add_computational_memlet(block, output_node_fma, min_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, one_node, min_op, "_in2", {}, scalar_type);
    auto& output_node_min = create_tmp_access_node(builder, block, "tmp_hs_min_", scalar_type);
    builder.add_computational_memlet(block, min_op, "_out", output_node_min, {}, scalar_type);

    // max(0, x)
    auto& zero_node = builder.add_constant(block, "0.0f", scalar_type);
    auto& last_op = builder.add_library_node<
        math::cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::fmax, input0.required_type);
    builder.add_computational_memlet(block, output_node_min, last_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, zero_node, last_op, "_in2", {}, scalar_type);

    return {.producer = &last_op, .output_conn_index = 0, .type = input0.required_type};
}


std::unique_ptr<data_flow::DataFlowNode> HardSigmoidNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new HardSigmoidNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
