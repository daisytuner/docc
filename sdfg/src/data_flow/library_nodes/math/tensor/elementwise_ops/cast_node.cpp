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
    types::PrimitiveType target_type
)
    : ElementWiseUnaryNode(element_id, debug_info, vertex, parent, LibraryNodeType_Cast, shape),
      target_type_(target_type) {}

bool CastNode::expand_operation(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& body,
    const std::string& input_name,
    const std::string& output_name,
    const types::Tensor& input_type,
    const types::Tensor& output_type,
    const data_flow::Subset& subset
) {
    // Add code block
    auto& code_block = builder.add_block(body);

    auto& input_node_new = builder.add_access(code_block, input_name);
    auto& output_node_new = builder.add_access(code_block, output_name);

    // Use assign tasklet which handles type casting when input and output types differ
    auto& tasklet = builder.add_tasklet(code_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(code_block, input_node_new, tasklet, "_in", subset, input_type);
    builder.add_computational_memlet(code_block, tasklet, "_out", output_node_new, subset, output_type);

    return true;
}

void CastNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // Check that all input memlets are tensor of scalar
    for (auto& iedge : graph.in_edges(*this)) {
        if (iedge.base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "CastNode: Input memlet must be of tensor type. Found type: " + iedge.base_type().print()
            );
        }
    }

    // Check that all output memlets are tensor of scalar
    for (auto& oedge : graph.out_edges(*this)) {
        if (oedge.base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "CastNode: Output memlet must be of tensor type. Found type: " + oedge.base_type().print()
            );
        }
    }

    // For CastNode, we DON'T check that all memlets have the same primitive type
    // because the whole point of casting is to convert between types
}

std::unique_ptr<data_flow::DataFlowNode> CastNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new CastNode(element_id, this->debug_info(), vertex, parent, this->shape_, this->target_type_)
    );
}

nlohmann::json CastNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const CastNode& cast_node = static_cast<const CastNode&>(library_node);
    nlohmann::json j;

    j["code"] = cast_node.code().value();

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (auto& dim : cast_node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    j["target_type"] = static_cast<int>(cast_node.target_type());

    return j;
}

data_flow::LibraryNode& CastNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("shape"));
    assert(j.contains("target_type"));

    std::vector<symbolic::Expression> shape;
    for (const auto& dim : j["shape"]) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    types::PrimitiveType target_type = static_cast<types::PrimitiveType>(j["target_type"].get<int>());

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return static_cast<CastNode&>(builder.add_library_node<CastNode>(parent, debug_info, shape, target_type));
}

} // namespace tensor
} // namespace math
} // namespace sdfg
