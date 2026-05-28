#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/mul_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

MulNode::MulNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Mul, shape, "C", {"A", "B"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput MulNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input0 = needed_inputs.at(0);

    data_flow::TaskletCode opcode;
    if (types::is_integer(input0.required_type)) {
        opcode = data_flow::TaskletCode::int_mul;
    } else {
        opcode = data_flow::TaskletCode::fp_mul;
    }
    auto& tasklet = builder.add_tasklet(block, opcode, "_out", {"_in1", "_in2"});
    input0.consumer = &tasklet;
    input0.input_conn_index = 0;
    needed_inputs.at(1).consumer = &tasklet;
    needed_inputs.at(1).input_conn_index = 1;
    return {.producer = &tasklet, .output_conn_index = 0, .type = input0.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> MulNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new MulNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
