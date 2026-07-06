#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/maximum_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

MaximumNode::MaximumNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Maximum, shape, "C", {"A", "B"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput MaximumNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input0 = needed_inputs.at(0);
    auto& input1 = needed_inputs.at(1);

    CodeNode* code_node;
    if (types::is_integer(input0.required_type)) {
        // Use tasklets for integer types - distinguish between signed and unsigned
        auto tasklet_code = TensorNode::get_integer_minmax_tasklet(input0.required_type, true);
        code_node = &builder.add_tasklet(block, tasklet_code, "_out", {"_in1", "_in2"});
    } else {
        // Use intrinsics for floating-point types with correct suffix
        code_node = &builder.add_library_node<
            cmath::CMathNode>(block, this->debug_info(), cmath::CMathFunction::fmax, input0.required_type);
    }

    input0.consumer = code_node;
    input0.input_conn_index = 0;
    input1.consumer = code_node;
    input1.input_conn_index = 1;

    return {.producer = code_node, .output_conn_index = 0, .type = input0.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> MaximumNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new MaximumNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
