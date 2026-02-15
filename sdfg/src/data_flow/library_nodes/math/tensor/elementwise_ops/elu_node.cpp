#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/elu_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace math {
namespace tensor {

EluNode::EluNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Elu, shape) {
    this->inputs_.push_back("alpha");
}

bool EluNode::expand_operation(
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
    auto& input_node = builder.add_access(code_block, input_name);
    auto& output_node_exp = builder.add_access(code_block, output_name);
    auto& output_node_sub = builder.add_access(code_block, output_name);
    auto& output_node_mul = builder.add_access(code_block, output_name);

    sdfg::types::Scalar element_type(output_type.primitive_type());

    // 1. exp(x)
    {
        auto& tasklet = builder.add_library_node<math::cmath::CMathNode>(
            code_block, code_block.debug_info(), cmath::CMathFunction::exp, input_type.primitive_type()
        );
        builder.add_computational_memlet(code_block, input_node, tasklet, "_in1", subset, input_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_exp, subset, output_type);
    }
    // 2. x - 1.0f
    {
        auto& one_node = builder.add_constant(code_block, "1.0", element_type);
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(code_block, output_node_exp, tasklet, "_in1", subset, output_type);
        builder.add_computational_memlet(code_block, one_node, tasklet, "_in2", {}, element_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_sub, subset, output_type);
    }
    // 3. alpha * x
    {
        auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(code_block, output_node_sub, tasklet, "_in1", subset, output_type);
        builder.add_computational_memlet(code_block, tasklet, "_out", output_node_mul, subset, output_type);

        // Find alpha node
        auto& graph = this->get_parent();
        const data_flow::Memlet* alpha_memlet = nullptr;
        for (auto& in_edge : graph.in_edges(*this)) {
            if (in_edge.dst_conn() == "alpha") {
                alpha_memlet = &in_edge;
                break;
            }
        }

        data_flow::AccessNode* alpha_node = nullptr;
        if (alpha_memlet) {
            auto& src = dynamic_cast<const data_flow::AccessNode&>(alpha_memlet->src());
            if (auto const_node = dynamic_cast<const data_flow::ConstantNode*>(&src)) {
                alpha_node = &builder.add_constant(code_block, const_node->data(), const_node->type());
            } else {
                alpha_node = &builder.add_access(code_block, src.data());
            }
            builder.add_computational_memlet(
                code_block, *alpha_node, tasklet, "_in2", alpha_memlet->subset(), alpha_memlet->base_type()
            );
        } else {
            alpha_node = &builder.add_constant(code_block, "1.0", element_type);
            builder.add_computational_memlet(code_block, *alpha_node, tasklet, "_in2", {}, element_type);
        }
    }

    return true;
}

std::unique_ptr<data_flow::DataFlowNode> EluNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new EluNode(element_id, this->debug_info(), vertex, parent, this->shape_));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
