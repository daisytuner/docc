#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/abs_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

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
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Abs, shape) {}

bool AbsNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::Tensor& input_type,
    const types::Tensor& output_type,
    const data_flow::Subset& subset
) {
    // Add code
    auto& code_block = builder.add_block(body);

    auto& input_node_new = builder.add_access(code_block, input_name);
    auto& output_node_new = builder.add_access(code_block, output_name);

    bool is_int = types::is_integer(output_type.primitive_type());

    if (is_int) {
        // For integers, use the int_abs tasklet
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::int_abs, "_out", {"_in"});
        builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", subset, input_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, subset, output_type);
    } else {
        // For floating-point, use the correct fabs intrinsic
        auto& libnode = builder.add_library_node<math::cmath::CMathNode>(
            code_block, body.debug_info(), cmath::CMathFunction::fabs, input_type.primitive_type()
        );
        builder.add_computational_memlet(code_block, input_node_new, libnode, "_in1", subset, input_type);
        builder.add_computational_memlet(code_block, libnode, "_out", output_node_new, subset, output_type);
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> AbsNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new AbsNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
