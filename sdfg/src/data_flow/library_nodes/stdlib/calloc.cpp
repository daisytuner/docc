#include "sdfg/data_flow/library_nodes/stdlib/calloc.h"

namespace sdfg {
namespace stdlib {

CallocNode::CallocNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression num,
    const symbolic::Expression size
)
    : StdlibNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Calloc,
          {"_ret"},
          {},
          true,
          data_flow::ImplementationType_NONE
      ),
      num_(num), size_(size) {}

const symbolic::Expression CallocNode::num() const { return num_; }

const symbolic::Expression CallocNode::size() const { return size_; }

void CallocNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet CallocNode::symbols() const {
    auto num_symbols = symbolic::atoms(this->num_);
    auto size_symbols = symbolic::atoms(this->size_);
    num_symbols.insert(size_symbols.begin(), size_symbols.end());
    return num_symbols;
}

std::unique_ptr<data_flow::DataFlowNode> CallocNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<CallocNode>(element_id, debug_info_, vertex, parent, num_, size_);
}

void CallocNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->size_ = symbolic::subs(this->size_, old_expression, new_expression);
    this->num_ = symbolic::subs(this->num_, old_expression, new_expression);
}

nlohmann::json CallocNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const CallocNode& node = static_cast<const CallocNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["size"] = serializer.expression(node.size());
    j["num"] = serializer.expression(node.num());

    return j;
}

data_flow::LibraryNode& CallocNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("size"));
    assert(j.contains("num"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Calloc.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Extract properties
    auto size = symbolic::parse(j.at("size"));
    auto num = symbolic::parse(j.at("num"));

    return builder.add_library_node<CallocNode>(parent, debug_info, num, size);
}

CallocNodeDispatcher::CallocNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const CallocNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void CallocNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& Calloc_node = static_cast<const CallocNode&>(node_);

    stream << Calloc_node.outputs().at(0);
    stream << " = ";
    stream << language_extension_.external_prefix() << "calloc(" << language_extension_.expression(Calloc_node.num())
           << ", " << language_extension_.expression(Calloc_node.size()) << ")" << ";";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
