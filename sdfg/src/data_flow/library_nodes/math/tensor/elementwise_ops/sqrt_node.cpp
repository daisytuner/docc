#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/sqrt_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

SqrtNode::SqrtNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Sqrt, shape, "Y", {"X"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput SqrtNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input = needed_inputs.at(0);

    auto& tasklet = builder.add_library_node<
        math::cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::sqrt, input.required_type);
    input.consumer = &tasklet;
    input.input_conn_index = 0;
    return {.producer = &tasklet, .output_conn_index = 0, .type = input.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> SqrtNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new SqrtNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
