#include "sdfg/data_flow/library_nodes/stdlib/trap.h"

namespace sdfg {
namespace stdlib {

TrapNode::TrapNode(
    size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
)
    : StdlibNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Trap, {}, {}, true, data_flow::ImplementationType_NONE
      ) {}

void TrapNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet TrapNode::symbols() const { return symbolic::SymbolSet(); }

std::unique_ptr<data_flow::DataFlowNode> TrapNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<TrapNode>(element_id, debug_info_, vertex, parent);
}

void TrapNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->size_ = symbolic::subs(this->size_, old_expression, new_expression);
}

nlohmann::json TrapNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const TrapNode& node = static_cast<const TrapNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    return j;
}

data_flow::LibraryNode& TrapNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Trap.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<TrapNode>(parent, debug_info);
}

TrapNodeDispatcher::TrapNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const TrapNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void TrapNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const TrapNode&>(node_);

    stream << "__builtin_trap()" << ";";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
