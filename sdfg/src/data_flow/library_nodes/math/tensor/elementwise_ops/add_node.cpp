#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/add_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/types/type.h"

#include "sdfg/analysis/scope_analysis.h"

namespace sdfg {
namespace math {
namespace tensor {

AddNode::AddNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseBinaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Add, shape) {}

bool AddNode::expand_operation(
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

    bool is_int = types::is_integer(output_type.primitive_type());
    data_flow::TaskletCode opcode = is_int ? data_flow::TaskletCode::int_add : data_flow::TaskletCode::fp_add;
    auto& tasklet = builder.add_tasklet(code_block, opcode, "_out", {"_in1", "_in2"});

    auto& output_node = builder.add_access(code_block, output_name);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node, subset, output_type);

    if (builder.subject().exists(input_name_a)) {
        auto& input_node_a = builder.add_access(code_block, input_name_a);
        if (input_type_a.is_scalar()) {
            builder.add_computational_memlet(code_block, input_node_a, tasklet, "_in1", {}, input_type_a);
        } else {
            builder.add_computational_memlet(code_block, input_node_a, tasklet, "_in1", subset, input_type_a);
        }
    } else {
        types::Scalar const_type(input_type_a.primitive_type());
        auto& input_node_a = builder.add_constant(code_block, input_name_a, const_type);
        builder.add_computational_memlet(code_block, input_node_a, tasklet, "_in1", subset, input_type_a);
    }

    if (builder.subject().exists(input_name_b)) {
        auto& input_node_b = builder.add_access(code_block, input_name_b);
        if (input_type_b.is_scalar()) {
            builder.add_computational_memlet(code_block, input_node_b, tasklet, "_in2", {}, input_type_b);
        } else {
            builder.add_computational_memlet(code_block, input_node_b, tasklet, "_in2", subset, input_type_b);
        }
    } else {
        types::Scalar const_type(input_type_b.primitive_type());
        auto& input_node_b = builder.add_constant(code_block, input_name_b, const_type);
        builder.add_computational_memlet(code_block, input_node_b, tasklet, "_in2", subset, input_type_b);
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> AddNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new AddNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
