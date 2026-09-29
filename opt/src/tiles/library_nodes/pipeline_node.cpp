#include "sdfg/tiles/library_nodes/pipeline_node.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/codegen/language_extensions/rocm_language_extension.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace tiles {

// ============================== PipelineCommitNode ===========================

PipelineCommitNode::PipelineCommitNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type
)
    : data_flow::LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_PipelineCommit, {}, {}, true, implementation_type
      ) {
}

void PipelineCommitNode::validate(const Function& function) const {
    data_flow::LibraryNode::validate(function);
}

symbolic::SymbolSet PipelineCommitNode::symbols() const {
    return {};
}

std::unique_ptr<data_flow::DataFlowNode> PipelineCommitNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<PipelineCommitNode>(
        new PipelineCommitNode(element_id, this->debug_info_, vertex, parent, this->implementation_type_)
    );
}
void PipelineCommitNode::replace(const symbolic::Expression, const symbolic::Expression) {
}

void PipelineCommitNode::replace(const symbolic::ExpressionMapping&) {
}

// ============================== PipelineWaitNode =============================

PipelineWaitNode::PipelineWaitNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    size_t keep_outstanding,
    size_t loads_per_group
)
    : data_flow::LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_PipelineWait, {}, {}, true, implementation_type
      ),
      keep_outstanding_(keep_outstanding), loads_per_group_(loads_per_group) {
}

void PipelineWaitNode::validate(const Function& function) const {
    data_flow::LibraryNode::validate(function);
}

symbolic::SymbolSet PipelineWaitNode::symbols() const {
    return {};
}

std::unique_ptr<data_flow::DataFlowNode> PipelineWaitNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<PipelineWaitNode>(new PipelineWaitNode(
        element_id, this->debug_info_, vertex, parent, this->implementation_type_, keep_outstanding_, loads_per_group_
    ));
}
void PipelineWaitNode::replace(const symbolic::Expression, const symbolic::Expression) {
}

void PipelineWaitNode::replace(const symbolic::ExpressionMapping&) {
}

// ============================== Serializers ==================================

nlohmann::json PipelineCommitNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    nlohmann::json j;
    j["code"] = std::string(library_node.code().value());
    return j;
}
data_flow::LibraryNode& PipelineCommitNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, sdfg::structured_control_flow::Block& parent
) {
    return builder
        .add_library_node<PipelineCommitNode>(parent, DebugInfo(), j.at("implementation_type").get<std::string>());
}

nlohmann::json PipelineWaitNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const PipelineWaitNode&>(library_node);
    nlohmann::json j;
    j["code"] = std::string(node.code().value());
    j["keep_outstanding"] = node.keep_outstanding();
    j["loads_per_group"] = node.loads_per_group();
    return j;
}
data_flow::LibraryNode& PipelineWaitNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, sdfg::structured_control_flow::Block& parent
) {
    return builder.add_library_node<PipelineWaitNode>(
        parent,
        DebugInfo(),
        j.at("implementation_type").get<std::string>(),
        j["keep_outstanding"].get<size_t>(),
        j.value("loads_per_group", static_cast<size_t>(1))
    );
}

} // namespace tiles
} // namespace sdfg
