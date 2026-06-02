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
          "X",
          {"Y", "alpha"},
          quantization,
          impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput LeakyReLUNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input0 = needed_inputs.at(0);
    auto& alpha_input = needed_inputs.at(1);

    throw std::runtime_error("LeakyReLUNode: untested expand");
    types::Scalar scalar_type(input0.required_type);

    // max(x, 0)
    auto& zero_node = builder.add_constant(block, "0.0", scalar_type);
    auto& first_op = builder.add_library_node<
        math::cmath::CMathNode>(block, block.debug_info(), cmath::CMathFunction::fmax, input0.required_type);
    input0.consumer = &first_op;
    input0.input_conn_index = 0;
    builder.add_computational_memlet(block, zero_node, first_op, "_in2", {}, scalar_type);
    auto& output_node_max = create_tmp_access_node(builder, block, "tmp_lkyrl_max_", scalar_type);
    builder.add_computational_memlet(block, first_op, "_out", output_node_max, {}, scalar_type);
    // alpha * x
    auto& mul_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    alpha_input.consumer = &mul_op;
    alpha_input.input_conn_index = 0;
    builder.add_computational_memlet(block, output_node_max, mul_op, "_in2", {}, scalar_type);

    return {.producer = &mul_op, .output_conn_index = 0, .type = input0.required_type};
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
