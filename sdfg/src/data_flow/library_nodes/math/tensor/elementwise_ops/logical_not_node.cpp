#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/logical_not_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

LogicalNotNode::LogicalNotNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_LogicalNot, shape, "Y", {"X"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput LogicalNotNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto input_type = needed_inputs.at(0).required_type;
    // baseline for logical_not is to use a tasklet with a comparison to 0, which will yield 1 for false and 0 for true
    auto& const_0 = builder.add_constant(block, "0", types::Scalar(input_type), this->debug_info());

    data_flow::TaskletCode tasklet_code;
    if (types::is_integer(input_type)) {
        tasklet_code = data_flow::TaskletCode::int_eq;
    } else if (types::is_floating_point(input_type)) {
        tasklet_code = data_flow::TaskletCode::fp_oeq;
    } else {
        throw InvalidSDFGException(&"LogicalNotNode: Unsupported expected type for expand_operation_dataflow: "
                                       [*types::primitive_type_to_string(input_type)]);
    }

    auto& tasklet = builder.add_tasklet(block, tasklet_code, "_out", {"_in0", "_in1"}, this->debug_info());
    builder.add_computational_memlet(block, const_0, tasklet, "_in1", {}, types::Scalar(input_type), this->debug_info());
    auto& input = needed_inputs.at(0);
    input.consumer = &tasklet;
    input.input_conn_index = 0;

    return {.producer = &tasklet, .output_conn_index = 0, .type = types::PrimitiveType::Bool};
}

void LogicalNotNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    validate_target_tensor(graph);

    validate_all_input_tensors(graph);

    validate_non_tensor_inputs(graph);
}

std::unique_ptr<data_flow::DataFlowNode> LogicalNotNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new LogicalNotNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
