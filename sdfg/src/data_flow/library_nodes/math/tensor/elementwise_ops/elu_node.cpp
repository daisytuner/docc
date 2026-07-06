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
      ) {}

ElementWiseDataflowTensorNode::ElementOutput EluNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input0 = needed_inputs.at(0);
    bool has_alpha_input = needed_inputs.size() > 1;

    types::Scalar scalar_type(input0.required_type);

    throw std::runtime_error("Elu: untested expand");

    // 1. exp(x)
    auto& first_op = builder.add_library_node<
        math::cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::exp, input0.required_type);
    input0.consumer = &first_op;
    input0.input_conn_index = 0;
    auto& output_node_exp = create_tmp_access_node(builder, block, "tmp_elu_exp_", scalar_type);
    builder.add_computational_memlet(block, first_op, "_out", output_node_exp, {}, scalar_type);
    // 2. x - 1.0f
    auto& one_node = builder.add_constant(block, "1.0", scalar_type);
    auto& sub_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, output_node_exp, sub_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, one_node, sub_op, "_in2", {}, scalar_type);
    auto& output_node_sub = create_tmp_access_node(builder, block, "tmp_elu_sub_", scalar_type);
    builder.add_computational_memlet(block, sub_op, "_out", output_node_sub, {}, scalar_type);
    // 3. alpha * x
    auto& last_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, output_node_sub, last_op, "_in1", {}, scalar_type);
    if (has_alpha_input) {
        auto& alpha_input = needed_inputs.at(1);
        alpha_input.consumer = &last_op;
        alpha_input.input_conn_index = 1;
    } else {
        builder.add_computational_memlet(block, one_node, last_op, "_in2", {}, scalar_type);
    }

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
