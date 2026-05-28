#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/fill_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

FillNode::FillNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Fill, shape, "Y", {"X"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput FillNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& input = needed_inputs.at(0);
    input.consumer = &tasklet;
    input.input_conn_index = 0;

    return {.producer = &tasklet, .output_conn_index = 0, .type = input.required_type};
}

std::unique_ptr<data_flow::DataFlowNode> FillNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new FillNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

void FillNode::validate_non_tensor_inputs(const data_flow::DataFlowGraph& graph) const {
    auto* fill_value_edge = graph.in_edge_for_connector(*this, inputs_.at(1));
    // Validate that the input is a scalar
    if (fill_value_edge && fill_value_edge->base_type().type_id() != types::TypeID::Scalar) {
        throw InvalidSDFGException(
            "FillNode: Input memlet must be of scalar type. Found type: " + fill_value_edge->base_type().print()
        );
    }
}

} // namespace tensor
} // namespace math
} // namespace sdfg
