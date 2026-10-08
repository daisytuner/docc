#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/elu_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

EluNode::EluNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Elu, shape, "Y", {"X", "alpha"}, quantization, impl_type
      ) {
}

ElementWiseDataflowTensorNode::ElementOutput EluNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input0 = needed_inputs.at(0);
    bool has_alpha_input = needed_inputs.size() > 1;

    types::Scalar scalar_type(input0.required_type);

    // elu(x) = max(x, 0) + alpha * expm1(min(x, 0))

    // x
    auto& assign_op = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    input0.consumer = &assign_op;
    input0.input_conn_index = 0;
    auto& output_node_x = create_tmp_access_node(builder, block, "tmp_elu_x_", scalar_type);
    builder.add_computational_memlet(block, assign_op, "_out", output_node_x, {}, scalar_type);

    // max(x, 0)
    auto& zero_node = builder.add_constant(block, "0.0", scalar_type);
    auto& max_op = builder.add_library_node<
        math::cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::fmax, input0.required_type);
    builder.add_computational_memlet(block, output_node_x, max_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, zero_node, max_op, "_in2", {}, scalar_type);
    auto& output_node_max = create_tmp_access_node(builder, block, "tmp_elu_max_", scalar_type);
    builder.add_computational_memlet(block, max_op, "_out", output_node_max, {}, scalar_type);

    // min(x, 0)
    auto& min_op = builder.add_library_node<
        math::cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::fmin, input0.required_type);
    builder.add_computational_memlet(block, output_node_x, min_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, zero_node, min_op, "_in2", {}, scalar_type);
    auto& output_node_min = create_tmp_access_node(builder, block, "tmp_elu_min_", scalar_type);
    builder.add_computational_memlet(block, min_op, "_out", output_node_min, {}, scalar_type);

    // expm1(min(x, 0)), avoids cancellation of exp(x) - 1 for small |x|
    auto& expm1_op = builder.add_library_node<
        math::cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::expm1, input0.required_type);
    builder.add_computational_memlet(block, output_node_min, expm1_op, "_in1", {}, scalar_type);
    auto& output_node_expm1 = create_tmp_access_node(builder, block, "tmp_elu_expm1_", scalar_type);
    builder.add_computational_memlet(block, expm1_op, "_out", output_node_expm1, {}, scalar_type);

    // alpha * expm1(min(x, 0)) + max(x, 0)
    auto& last_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
    if (has_alpha_input) {
        auto& alpha_input = needed_inputs.at(1);
        alpha_input.consumer = &last_op;
        alpha_input.input_conn_index = 0;
    } else {
        auto& one_node = builder.add_constant(block, "1.0", scalar_type);
        builder.add_computational_memlet(block, one_node, last_op, "_in1", {}, scalar_type);
    }
    builder.add_computational_memlet(block, output_node_expm1, last_op, "_in2", {}, scalar_type);
    builder.add_computational_memlet(block, output_node_max, last_op, "_in3", {}, scalar_type);

    return {.producer = &last_op, .output_conn_index = 0, .type = input0.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> EluNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new EluNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
