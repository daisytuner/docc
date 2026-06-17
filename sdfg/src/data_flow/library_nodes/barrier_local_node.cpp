#include "sdfg/data_flow/library_nodes/barrier_local_node.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/codegen/language_extensions/rocm_language_extension.h"

namespace sdfg {
namespace data_flow {

BarrierLocalNode::
    BarrierLocalNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, DataFlowGraph& parent)
    : LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_BarrierLocal, {}, {}, true, ImplementationType_NONE
      ) {

      };

void BarrierLocalNode::validate(const Function& function) const {
    LibraryNode::validate(function);
    // No specific validation for barrier local
}

symbolic::SymbolSet BarrierLocalNode::symbols() const { return {}; };

std::unique_ptr<DataFlowNode> BarrierLocalNode::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::unique_ptr<BarrierLocalNode>(new BarrierLocalNode(element_id, this->debug_info_, vertex, parent));
};

void BarrierLocalNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    // Do nothing
};

void BarrierLocalNode::replace(const symbolic::ExpressionMapping& replacements) {
    // Do nothing
};

nlohmann::json BarrierLocalNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    if (library_node.code() != data_flow::LibraryNodeType_BarrierLocal) {
        throw std::runtime_error("Invalid library node code");
    }
    nlohmann::json j;
    j["code"] = std::string(library_node.code().value());
    return j;
}

data_flow::LibraryNode& BarrierLocalNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, sdfg::structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != data_flow::LibraryNodeType_BarrierLocal.value()) {
        throw std::runtime_error("Invalid library node code");
    }
    return builder.add_library_node<data_flow::BarrierLocalNode>(parent, DebugInfo());
};

BarrierLocalNodeDispatcher::BarrierLocalNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const BarrierLocalNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BarrierLocalNodeDispatcher::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    if (dynamic_cast<codegen::CLanguageExtension*>(&this->language_extension_) != nullptr) {
        throw std::runtime_error(
            "ThreadBarrierDispatcher is not supported for C language extension. Use CUDA or ROCM "
            "language extension instead."
        );
    } else if (dynamic_cast<codegen::CPPLanguageExtension*>(&this->language_extension_) != nullptr) {
        throw std::runtime_error(
            "ThreadBarrierDispatcher is not supported for C++ language extension. Use CUDA or ROCM "
            "language extension instead."
        );
    } else if (dynamic_cast<codegen::CUDALanguageExtension*>(&this->language_extension_) != nullptr) {
        stream << "__syncthreads();" << std::endl;
    } else if (dynamic_cast<codegen::ROCMLanguageExtension*>(&this->language_extension_) != nullptr) {
        stream << "__syncthreads();" << std::endl;
    } else {
        throw std::runtime_error("Unsupported language extension for ThreadBarrierDispatcher");
    }
}


} // namespace data_flow
} // namespace sdfg
