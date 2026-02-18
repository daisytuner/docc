#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/min_node.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

MinNode::MinNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : ReduceNode(element_id, debug_info, vertex, parent, LibraryNodeType_Min, shape, axes, keepdims) {}

bool MinNode::expand_reduction(
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

    auto& in_access = builder.add_access(block, input_name, this->debug_info());
    auto& out_read_access = builder.add_access(block, output_name, this->debug_info());
    auto& out_write_access = builder.add_access(block, output_name, this->debug_info());

    bool is_int = types::is_integer(input_type.primitive_type());

    if (is_int) {
        // For integers, use tasklet - distinguish between signed and unsigned
        auto tasklet_code = TensorNode::get_integer_minmax_tasklet(input_type.primitive_type(), false);
        auto& tasklet = builder.add_tasklet(block, tasklet_code, "_out", {"_in1", "_in2"});

        builder
            .add_computational_memlet(block, in_access, tasklet, "_in1", input_subset, input_type, this->debug_info());
        builder.add_computational_memlet(
            block, out_read_access, tasklet, "_in2", output_subset, output_type, this->debug_info()
        );
        builder.add_computational_memlet(
            block, tasklet, "_out", out_write_access, output_subset, output_type, this->debug_info()
        );
    } else {
        // For floating-point, use the correct fmin intrinsic
        auto& libnode = builder.add_library_node<
            math::cmath::CMathNode>(block, this->debug_info(), cmath::CMathFunction::fmin, input_type.primitive_type());

        builder
            .add_computational_memlet(block, in_access, libnode, "_in1", input_subset, input_type, this->debug_info());
        builder.add_computational_memlet(
            block, out_read_access, libnode, "_in2", output_subset, output_type, this->debug_info()
        );
        builder.add_computational_memlet(
            block, libnode, "_out", out_write_access, output_subset, output_type, this->debug_info()
        );
    }

    return true;
}

std::string MinNode::identity(types::PrimitiveType primitive_type) const {
    switch (primitive_type) {
        case types::PrimitiveType::Int8:
            return "INT8_MAX";
        case types::PrimitiveType::Int16:
            return "INT16_MAX";
        case types::PrimitiveType::Int32:
            return "INT32_MAX";
        case types::PrimitiveType::Int64:
            return "INT64_MAX";
        case types::PrimitiveType::UInt8:
            return "UINT8_MAX";
        case types::PrimitiveType::UInt16:
            return "UINT16_MAX";
        case types::PrimitiveType::UInt32:
            return "UINT32_MAX";
        case types::PrimitiveType::UInt64:
            return "UINT64_MAX";
        default:
            return "INFINITY";
    }
}

std::unique_ptr<data_flow::DataFlowNode> MinNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new MinNode(element_id, this->debug_info(), vertex, parent, this->shape_, this->axes_, this->keepdims_)
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg
