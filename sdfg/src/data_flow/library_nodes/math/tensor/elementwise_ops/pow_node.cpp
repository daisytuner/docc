#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/pow_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

PowNode::PowNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Pow, shape, "C", {"A", "B"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput PowNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input0 = needed_inputs.at(0);
    auto& input1 = needed_inputs.at(1);

    auto& libnode =
        builder.add_library_node<cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::pow, input0.required_type);
    input0.consumer = &libnode;
    input0.input_conn_index = 0;
    input1.consumer = &libnode;
    input1.input_conn_index = 1;
    return {.producer = &libnode, .output_conn_index = 0, .type = input0.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> PowNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new PowNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
