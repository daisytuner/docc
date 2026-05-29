#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/cast_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace tensor {

CastNode::CastNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    types::PrimitiveType target_type,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Cast, shape, "Y", {"X"}, target_type, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput CastNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input = needed_inputs.at(0);

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    input.consumer = &tasklet;
    input.input_conn_index = 0;
    return {.producer = &tasklet, .output_conn_index = 0, .type = expected_type};
}

void CastNode::validate(const Function& function) const {
    // For CastNode, skip TensorNode validation, as that includes checking that primitive types match across everything!
    // because the whole point of casting is to convert between types
    MathNode::validate(function);
    auto& graph = this->get_parent();

    validate_all_input_tensors(graph);

    if (fixed_quantization_ != QUANTIZATION_MATCH_INPUTS) {
        auto* result = graph.in_edge_for_connector(*this, inputs_.at(0));
        if (!result) {
            throw InvalidSDFGException("CastNode # " + std::to_string(element_id_) + ": result tensor is not connected");
        }
        if (result->base_type().primitive_type() != fixed_quantization_) {
            throw InvalidSDFGException(
                "CastNode #" + std::to_string(element_id_) + ": result tensor has wrong primitive type. Expected " +
                types::primitive_type_to_string(fixed_quantization_) + ", got " +
                types::primitive_type_to_string(result->base_type().primitive_type())
            );
        }
    }
}

std::unique_ptr<data_flow::DataFlowNode> CastNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new CastNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
