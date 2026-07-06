#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

ReLUNode::ReLUNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_ReLU, shape, "Y", {"X"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput ReLUNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input = needed_inputs.at(0);

    types::Scalar zero_type(input.required_type);
    auto& zero_node = builder.add_constant(block, "0.0", zero_type);

    auto& libnode =
        builder.add_library_node<cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::fmax, input.required_type);
    input.consumer = &libnode;
    input.input_conn_index = 0;

    builder.add_computational_memlet(block, zero_node, libnode, "_in2", {}, zero_type);

    return {.producer = &libnode, .output_conn_index = 0, .type = input.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> ReLUNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new ReLUNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
