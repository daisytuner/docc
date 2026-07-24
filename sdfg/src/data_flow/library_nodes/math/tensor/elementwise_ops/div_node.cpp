#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/div_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

DivNode::DivNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Div, shape, "C", {"A", "B"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput DivNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input0 = needed_inputs.at(0);

    data_flow::TaskletCode opcode;
    if (types::is_integer(input0.required_type)) {
        bool is_signed = types::is_signed(input0.required_type);
        opcode = is_signed ? data_flow::TaskletCode::int_sdiv : data_flow::TaskletCode::int_udiv;
    } else {
        opcode = data_flow::TaskletCode::fp_div;
    }
    auto& tasklet = builder.add_tasklet(block, opcode, "_out", {"_in1", "_in2"});
    input0.consumer = &tasklet;
    input0.input_conn_index = 0;
    needed_inputs.at(1).consumer = &tasklet;
    needed_inputs.at(1).input_conn_index = 1;
    return {.producer = &tasklet, .output_conn_index = 0, .type = input0.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> DivNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new DivNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
