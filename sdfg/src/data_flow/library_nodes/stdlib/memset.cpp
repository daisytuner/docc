#include "sdfg/data_flow/library_nodes/stdlib/memset.h"

namespace sdfg {
namespace stdlib {

MemsetNode::MemsetNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression value,
    const symbolic::Expression num
)
    : StdlibNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Memset,
          {"_ptr"},
          {},
          true,
          data_flow::ImplementationType_NONE
      ),
      num_(num), value_(value) {}

const symbolic::Expression MemsetNode::value() const { return value_; }

const symbolic::Expression MemsetNode::num() const { return num_; }

void MemsetNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet MemsetNode::symbols() const {
    auto value_symbols = symbolic::atoms(this->value_);
    auto num_symbols = symbolic::atoms(this->num_);
    num_symbols.insert(value_symbols.begin(), value_symbols.end());
    return num_symbols;
}

std::unique_ptr<data_flow::DataFlowNode> MemsetNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<MemsetNode>(element_id, debug_info_, vertex, parent, value_, num_);
}

void MemsetNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->value_ = symbolic::subs(this->value_, old_expression, new_expression);
    this->num_ = symbolic::subs(this->num_, old_expression, new_expression);
}

nlohmann::json MemsetNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MemsetNode& node = static_cast<const MemsetNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["value"] = serializer.expression(node.value());
    j["num"] = serializer.expression(node.num());

    return j;
}

data_flow::LibraryNode& MemsetNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("value"));
    assert(j.contains("num"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Memset.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Extract properties
    auto value = symbolic::parse(j.at("value"));
    auto num = symbolic::parse(j.at("num"));

    return builder.add_library_node<MemsetNode>(parent, debug_info, value, num);
}

MemsetNodeDispatcher::MemsetNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const MemsetNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemsetNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const MemsetNode&>(node_);

    stream << language_extension_.external_prefix() << "memset(" << node.outputs().at(0) << ", "
           << language_extension_.expression(node.value()) << ", " << language_extension_.expression(node.num()) << ")"
           << ";";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
