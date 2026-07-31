#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/rsqrt_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

RsqrtNode::RsqrtNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Rsqrt, shape, "Y", {"X"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput RsqrtNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input = needed_inputs.at(0);
    types::Scalar scalar_type(input.required_type);

    // sqrt(x)
    auto& sqrt_op = builder.add_library_node<
        math::cmath::CMathNode>(block, block.debug_info(), cmath::CMathFunction::sqrt, input.required_type);
    input.consumer = &sqrt_op;
    input.input_conn_index = 0;

    auto& output_node_sqrt = create_tmp_access_node(builder, block, "tmp_rsqrt_sqrt_", scalar_type);
    builder.add_computational_memlet(block, sqrt_op, "_out", output_node_sqrt, {}, scalar_type);

    // 1.0 constant
    auto& one_node = builder.add_constant(block, "1.0", scalar_type);

    // 1.0 / sqrt(x)
    auto& div_op = builder.add_tasklet(block, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, one_node, div_op, "_in1", {}, scalar_type);
    builder.add_computational_memlet(block, output_node_sqrt, div_op, "_in2", {}, scalar_type);

    return {.producer = &div_op, .output_conn_index = 0, .type = input.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> RsqrtNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new RsqrtNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
