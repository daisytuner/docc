#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/abs_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

AbsNode::AbsNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : ElementWiseDataflowTensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Abs, shape, "Y", {"X"}, quantization, impl_type
      ) {}

ElementWiseDataflowTensorNode::ElementOutput AbsNode::expand_operation_dataflow(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    Block& block,
    std::vector<ElementInput>& needed_inputs,
    types::PrimitiveType expected_type
) {
    auto& input = needed_inputs.at(0);

    if (types::is_integer(input.required_type)) {
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_abs, "_out", {"_in"});
        input.consumer = &tasklet;
        input.input_conn_index = 0;
        return {.producer = &tasklet, .output_conn_index = 0, .type = input.required_type};
    } else {
        auto& libnode = builder.add_library_node<
            math::cmath::CMathNode>(block, debug_info_, cmath::CMathFunction::fabs, input.required_type);
        input.consumer = &libnode;
        input.input_conn_index = 0;
        return {.producer = &libnode, .output_conn_index = 0, .type = input.required_type};
    }
}

std::unique_ptr<data_flow::DataFlowNode> AbsNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new AbsNode(
        element_id, this->debug_info(), vertex, parent, this->shape_, fixed_quantization_, implementation_type_
    ));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
