#include "sdfg/data_flow/library_nodes/stdlib/unreachable.h"

namespace sdfg {
namespace stdlib {

UnreachableNode::UnreachableNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : StdlibNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Unreachable,
          {},
          {},
          true,
          data_flow::ImplementationType_NONE
      ) {}

void UnreachableNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet UnreachableNode::symbols() const { return symbolic::SymbolSet(); }

std::unique_ptr<data_flow::DataFlowNode> UnreachableNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<UnreachableNode>(element_id, debug_info_, vertex, parent);
}

void UnreachableNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->size_ = symbolic::subs(this->size_, old_expression, new_expression);
}

nlohmann::json UnreachableNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const UnreachableNode& node = static_cast<const UnreachableNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    return j;
}

data_flow::LibraryNode& UnreachableNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Unreachable.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<UnreachableNode>(parent, debug_info);
}

UnreachableNodeDispatcher::UnreachableNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const UnreachableNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void UnreachableNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const UnreachableNode&>(node_);

    stream << "__builtin_unreachable()" << ";";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
