#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/sum_node.h"

#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace tensor {

SumNode::SumNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : ReduceNode(element_id, debug_info, vertex, parent, LibraryNodeType_Sum, shape, axes, keepdims) {}

bool SumNode::expand_reduction(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::Tensor& input_type,
    const types::Tensor& output_type,
    const data_flow::Subset& input_subset,
    const data_flow::Subset& output_subset
) {
    auto& block = builder.add_block(body, {}, this->debug_info());

    bool is_int = types::is_integer(output_type.primitive_type());
    data_flow::TaskletCode opcode = is_int ? data_flow::TaskletCode::int_add : data_flow::TaskletCode::fp_add;

    auto& tasklet = builder.add_tasklet(block, opcode, {"_out"}, {"_in1", "_in2"}, this->debug_info());

    auto& in_access = builder.add_access(block, input_name, this->debug_info());
    auto& out_read_access = builder.add_access(block, output_name, this->debug_info());
    auto& out_write_access = builder.add_access(block, output_name, this->debug_info());

    builder.add_computational_memlet(block, in_access, tasklet, "_in1", input_subset, input_type, this->debug_info());
    builder
        .add_computational_memlet(block, out_read_access, tasklet, "_in2", output_subset, output_type, this->debug_info());
    builder.add_computational_memlet(
        block, tasklet, "_out", out_write_access, output_subset, output_type, this->debug_info()
    );

    return true;
}

std::string SumNode::identity(types::PrimitiveType primitive_type) const {
    if (types::is_integer(primitive_type)) {
        return "0";
    }
    return "0.0";
}

std::unique_ptr<data_flow::DataFlowNode> SumNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<
        SumNode>(element_id, this->debug_info(), vertex, parent, this->shape_, this->axes_, this->keepdims_);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
