#include "sdfg/tiles/library_nodes/async_copy_node.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/codegen/language_extensions/rocm_language_extension.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace tiles {

// ============================== CpAsyncCopyNode ==============================

CpAsyncCopyNode::CpAsyncCopyNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    size_t bytes
)
    : data_flow::LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_CpAsyncCopy,
          {},
          {"_dst", "_src"},
          true,
          implementation_type
      ),
      bytes_(bytes) {}

void CpAsyncCopyNode::validate(const Function& function) const { data_flow::LibraryNode::validate(function); }

symbolic::SymbolSet CpAsyncCopyNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> CpAsyncCopyNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<CpAsyncCopyNode>(
        new CpAsyncCopyNode(element_id, this->debug_info_, vertex, parent, this->implementation_type_, bytes_)
    );
}

void CpAsyncCopyNode::replace(const symbolic::Expression, const symbolic::Expression) {}

void CpAsyncCopyNode::replace(const symbolic::ExpressionMapping&) {}

data_flow::PointerAccessType CpAsyncCopyNode::pointer_access_type(int input_idx) const {
    auto size = symbolic::integer(static_cast<long long>(bytes_));
    if (input_idx == 0) { // _dst
        return data_flow::PointerAccessMeta::create_full_write_only(size, /*no_capture=*/true);
    }
    if (input_idx == 1) { // _src
        return data_flow::PointerAccessMeta::create_read_only(size, /*no_capture=*/true);
    }
    return data_flow::LibraryNode::pointer_access_type(input_idx);
}

// ============================== VectorCopyNode ==============================

VectorCopyNode::VectorCopyNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    size_t bytes
)
    : data_flow::LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_VectorCopy,
          {},
          {"_dst", "_src"},
          true,
          implementation_type
      ),
      bytes_(bytes) {}

void VectorCopyNode::validate(const Function& function) const { data_flow::LibraryNode::validate(function); }

symbolic::SymbolSet VectorCopyNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> VectorCopyNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<VectorCopyNode>(
        new VectorCopyNode(element_id, this->debug_info_, vertex, parent, this->implementation_type_, bytes_)
    );
}

void VectorCopyNode::replace(const symbolic::Expression, const symbolic::Expression) {}

void VectorCopyNode::replace(const symbolic::ExpressionMapping&) {}

data_flow::PointerAccessType VectorCopyNode::pointer_access_type(int input_idx) const {
    auto size = symbolic::integer(static_cast<long long>(bytes_));
    if (input_idx == 0) { // _dst
        return data_flow::PointerAccessMeta::create_full_write_only(size, /*no_capture=*/true);
    }
    if (input_idx == 1) { // _src
        return data_flow::PointerAccessMeta::create_read_only(size, /*no_capture=*/true);
    }
    return data_flow::LibraryNode::pointer_access_type(input_idx);
}

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
      ) {}

void PipelineCommitNode::validate(const Function& function) const { data_flow::LibraryNode::validate(function); }

symbolic::SymbolSet PipelineCommitNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> PipelineCommitNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<PipelineCommitNode>(
        new PipelineCommitNode(element_id, this->debug_info_, vertex, parent, this->implementation_type_)
    );
}
void PipelineCommitNode::replace(const symbolic::Expression, const symbolic::Expression) {}

void PipelineCommitNode::replace(const symbolic::ExpressionMapping&) {}

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
      keep_outstanding_(keep_outstanding), loads_per_group_(loads_per_group) {}

void PipelineWaitNode::validate(const Function& function) const { data_flow::LibraryNode::validate(function); }

symbolic::SymbolSet PipelineWaitNode::symbols() const { return {}; }

std::unique_ptr<data_flow::DataFlowNode> PipelineWaitNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<PipelineWaitNode>(new PipelineWaitNode(
        element_id, this->debug_info_, vertex, parent, this->implementation_type_, keep_outstanding_, loads_per_group_
    ));
}
void PipelineWaitNode::replace(const symbolic::Expression, const symbolic::Expression) {}

void PipelineWaitNode::replace(const symbolic::ExpressionMapping&) {}

// ============================== Serializers ==================================

nlohmann::json CpAsyncCopyNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const CpAsyncCopyNode&>(library_node);
    nlohmann::json j;
    j["code"] = std::string(node.code().value());
    j["bytes"] = node.bytes();
    return j;
}
data_flow::LibraryNode& CpAsyncCopyNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, sdfg::structured_control_flow::Block& parent
) {
    return builder.add_library_node<
        CpAsyncCopyNode>(parent, DebugInfo(), j.at("implementation_type").get<std::string>(), j["bytes"].get<size_t>());
}

nlohmann::json VectorCopyNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const VectorCopyNode&>(library_node);
    nlohmann::json j;
    j["code"] = std::string(node.code().value());
    j["bytes"] = node.bytes();
    return j;
}
data_flow::LibraryNode& VectorCopyNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, sdfg::structured_control_flow::Block& parent
) {
    return builder.add_library_node<
        VectorCopyNode>(parent, DebugInfo(), j.at("implementation_type").get<std::string>(), j["bytes"].get<size_t>());
}

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
