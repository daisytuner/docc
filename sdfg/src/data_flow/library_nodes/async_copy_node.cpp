#include "sdfg/data_flow/library_nodes/async_copy_node.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/codegen/language_extensions/rocm_language_extension.h"

namespace sdfg {
namespace data_flow {

namespace {
bool is_cuda(const codegen::LanguageExtension& le) {
    return dynamic_cast<const codegen::CUDALanguageExtension*>(&le) != nullptr;
}
bool is_rocm(const codegen::LanguageExtension& le) {
    return dynamic_cast<const codegen::ROCMLanguageExtension*>(&le) != nullptr;
}

// Preprocessor guard selecting the CDNA archs (gfx9xx) that support the
// asynchronous direct global->LDS load path (`global_load_lds` / vmcnt). RDNA
// (gfx10xx/11xx/12xx) lacks it, so the emitted #else keeps a synchronous copy.
constexpr const char* kCdnaArchGuard =
    "defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || "
    "defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)";
} // namespace

// ============================== CpAsyncCopyNode ==============================

CpAsyncCopyNode::CpAsyncCopyNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent, size_t bytes
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_CpAsyncCopy,
          {},
          {"_dst", "_src"},
          true,
          ImplementationType_NONE
      ),
      bytes_(bytes) {}

void CpAsyncCopyNode::validate(const Function& function) const { LibraryNode::validate(function); }
symbolic::SymbolSet CpAsyncCopyNode::symbols() const { return {}; }
std::unique_ptr<DataFlowNode> CpAsyncCopyNode::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::unique_ptr<CpAsyncCopyNode>(new CpAsyncCopyNode(element_id, this->debug_info_, vertex, parent, bytes_));
}
void CpAsyncCopyNode::replace(const symbolic::Expression, const symbolic::Expression) {}
void CpAsyncCopyNode::replace(const symbolic::ExpressionMapping&) {}

// ============================== PipelineCommitNode ===========================

PipelineCommitNode::PipelineCommitNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent
)
    : LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_PipelineCommit, {}, {}, true, ImplementationType_NONE
      ) {}

void PipelineCommitNode::validate(const Function& function) const { LibraryNode::validate(function); }
symbolic::SymbolSet PipelineCommitNode::symbols() const { return {}; }
std::unique_ptr<DataFlowNode> PipelineCommitNode::
    clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent) const {
    return std::unique_ptr<PipelineCommitNode>(new PipelineCommitNode(element_id, this->debug_info_, vertex, parent));
}
void PipelineCommitNode::replace(const symbolic::Expression, const symbolic::Expression) {}
void PipelineCommitNode::replace(const symbolic::ExpressionMapping&) {}

// ============================== PipelineWaitNode =============================

PipelineWaitNode::PipelineWaitNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    size_t keep_outstanding,
    size_t loads_per_group
)
    : LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_PipelineWait, {}, {}, true, ImplementationType_NONE
      ),
      keep_outstanding_(keep_outstanding), loads_per_group_(loads_per_group) {}

void PipelineWaitNode::validate(const Function& function) const { LibraryNode::validate(function); }
symbolic::SymbolSet PipelineWaitNode::symbols() const { return {}; }
std::unique_ptr<DataFlowNode> PipelineWaitNode::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::unique_ptr<PipelineWaitNode>(
        new PipelineWaitNode(element_id, this->debug_info_, vertex, parent, keep_outstanding_, loads_per_group_)
    );
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
    return builder.add_library_node<data_flow::CpAsyncCopyNode>(parent, DebugInfo(), j["bytes"].get<size_t>());
}

nlohmann::json PipelineCommitNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    nlohmann::json j;
    j["code"] = std::string(library_node.code().value());
    return j;
}
data_flow::LibraryNode& PipelineCommitNodeSerializer::deserialize(
    const nlohmann::json&, sdfg::builder::StructuredSDFGBuilder& builder, sdfg::structured_control_flow::Block& parent
) {
    return builder.add_library_node<data_flow::PipelineCommitNode>(parent, DebugInfo());
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
    return builder.add_library_node<data_flow::PipelineWaitNode>(
        parent, DebugInfo(), j["keep_outstanding"].get<size_t>(), j.value("loads_per_group", static_cast<size_t>(1))
    );
}

// ============================== Dispatchers ==================================

CpAsyncCopyNodeDispatcher::CpAsyncCopyNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::CpAsyncCopyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void CpAsyncCopyNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    const auto& node = static_cast<const CpAsyncCopyNode&>(node_);
    // Connector order {"_dst", "_src"} — both are addresses (reference memlets).
    const std::string& dst = inputs.at(0).expr;
    const std::string& src = inputs.at(1).expr;
    const size_t bytes = node.bytes();

    const size_t words = bytes / 4;
    if (is_cuda(language_extension_)) {
        out.stream << "__pipeline_memcpy_async(" << dst << ", " << src << ", " << bytes << ");" << std::endl;
    } else if (is_rocm(language_extension_)) {
        // CDNA has an asynchronous direct global->LDS load; RDNA does not.
        out.stream << "#if " << kCdnaArchGuard << std::endl;
        // CDNA: per-word async global->LDS load (in flight, tracked by vmcnt,
        // drained by the matching PipelineWait). dst points into __shared__.
        out.stream << "for (size_t __i = 0; __i < " << words << "; ++__i) "
                   << "__builtin_amdgcn_global_load_lds(reinterpret_cast<const unsigned*>(" << src
                   << ") + __i, reinterpret_cast<unsigned*>(" << dst << ") + __i, 4, 0, 0);" << std::endl;
        out.stream << "#else" << std::endl;
        // RDNA (and any non-CDNA): no async LDS path — copy synchronously.
        out.stream << "for (size_t __i = 0; __i < " << words << "; ++__i) "
                   << "reinterpret_cast<float*>(" << dst << ")[__i] = reinterpret_cast<const float*>(" << src
                   << ")[__i];" << std::endl;
        out.stream << "#endif" << std::endl;
    } else {
        throw std::runtime_error("CpAsyncCopyNode requires a CUDA or ROCM language extension");
    }
}

PipelineCommitNodeDispatcher::PipelineCommitNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::PipelineCommitNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void PipelineCommitNodeDispatcher::
    dispatch(codegen::PrettyPrinter& stream, codegen::PrettyPrinter&, codegen::CodeSnippetFactory&) {
    if (is_cuda(language_extension_)) {
        stream << "__pipeline_commit();" << std::endl;
    } else if (is_rocm(language_extension_)) {
        // No commit primitive on either arch: CDNA loads are tracked directly by
        // the hardware vmcnt (drained in PipelineWait); RDNA is synchronous.
    } else {
        throw std::runtime_error("PipelineCommitNode requires a CUDA or ROCM language extension");
    }
}

PipelineWaitNodeDispatcher::PipelineWaitNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::PipelineWaitNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void PipelineWaitNodeDispatcher::
    dispatch(codegen::PrettyPrinter& stream, codegen::PrettyPrinter&, codegen::CodeSnippetFactory&) {
    const auto& node = static_cast<const PipelineWaitNode&>(node_);
    if (is_cuda(language_extension_)) {
        stream << "__pipeline_wait_prior(" << node.keep_outstanding() << ");" << std::endl;
    } else if (is_rocm(language_extension_)) {
        // CDNA: drain direct-to-LDS loads on the flat vmcnt counter, keeping
        // keep_outstanding pipeline stages (each loads_per_group words) still in
        // flight. RDNA: synchronous, nothing outstanding to wait on.
        stream << "#if " << kCdnaArchGuard << std::endl;
        stream << "asm volatile(\"s_waitcnt vmcnt(" << (node.keep_outstanding() * node.loads_per_group()) << ")\");"
               << std::endl;
        stream << "#endif" << std::endl;
    } else {
        throw std::runtime_error("PipelineWaitNode requires a CUDA or ROCM language extension");
    }
}

} // namespace data_flow
} // namespace sdfg
