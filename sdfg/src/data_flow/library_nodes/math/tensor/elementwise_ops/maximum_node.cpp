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
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseBinaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Maximum, shape) {}

bool MaximumNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name_a,
    const std::string& input_name_b,
    const std::string& output_name,
    const types::Tensor& input_type_a,
    const types::Tensor& input_type_b,
    const types::Tensor& output_type,
    const data_flow::Subset& subset
) {
    auto& code_block = builder.add_block(body);
    data_flow::AccessNode* input_node_a;
    if (builder.subject().exists(input_name_a)) {
        input_node_a = &builder.add_access(code_block, input_name_a);
    } else {
        types::Scalar const_type(input_type_a.primitive_type());
        input_node_a = &builder.add_constant(code_block, input_name_a, const_type);
    }
    data_flow::AccessNode* input_node_b;
    if (builder.subject().exists(input_name_b)) {
        input_node_b = &builder.add_access(code_block, input_name_b);
    } else {
        types::Scalar const_type(input_type_b.primitive_type());
        input_node_b = &builder.add_constant(code_block, input_name_b, const_type);
    }
    auto& output_node = builder.add_access(code_block, output_name);

    bool is_int = types::is_integer(input_type_a.primitive_type());

    if (is_int) {
        // Use tasklets for integer types - distinguish between signed and unsigned
        auto tasklet_code = TensorNode::get_integer_minmax_tasklet(input_type_a.primitive_type(), true);
        auto& tasklet = builder.add_tasklet(code_block, tasklet_code, "_out", {"_in1", "_in2"});

        builder.add_computational_memlet(code_block, *input_node_a, tasklet, "_in1", subset, input_type_a);
        builder.add_computational_memlet(code_block, *input_node_b, tasklet, "_in2", subset, input_type_b);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node, subset, output_type);
    } else {
        // Use intrinsics for floating-point types with correct suffix
        auto& node = builder.add_library_node<
            cmath::CMathNode>(code_block, this->debug_info(), cmath::CMathFunction::fmax, input_type_a.primitive_type());

        builder.add_computational_memlet(code_block, *input_node_a, node, "_in1", subset, input_type_a, DebugInfo());
        builder.add_computational_memlet(code_block, *input_node_b, node, "_in2", subset, input_type_b, DebugInfo());
        builder.add_computational_memlet(code_block, node, "_out", output_node, subset, output_type, DebugInfo());
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> MaximumNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new MaximumNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
