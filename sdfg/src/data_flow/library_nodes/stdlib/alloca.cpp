#include "sdfg/data_flow/library_nodes/stdlib/alloca.h"

namespace sdfg {
namespace stdlib {

AllocaNode::AllocaNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression size
)
    : StdlibNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Alloca,
          {"_ret"},
          {},
          true,
          data_flow::ImplementationType_NONE
      ),
      size_(size) {}

const symbolic::Expression AllocaNode::size() const { return size_; }

void AllocaNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet AllocaNode::symbols() const { return symbolic::atoms(this->size_); }

std::unique_ptr<data_flow::DataFlowNode> AllocaNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<AllocaNode>(element_id, debug_info_, vertex, parent, size_);
}

void AllocaNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->size_ = symbolic::subs(this->size_, old_expression, new_expression);
}

nlohmann::json AllocaNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const AllocaNode& node = static_cast<const AllocaNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["size"] = serializer.expression(node.size());

    return j;
}

data_flow::LibraryNode& AllocaNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("size"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Alloca.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto size = symbolic::parse(j.at("size"));

    return builder.add_library_node<AllocaNode>(parent, debug_info, size);
}

AllocaNodeDispatcher::AllocaNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const AllocaNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void AllocaNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const AllocaNode&>(node_);

    stream << node.outputs().at(0);
    stream << " = ";
    stream << "alloca(" << language_extension_.expression(node.size()) << ")" << ";";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
