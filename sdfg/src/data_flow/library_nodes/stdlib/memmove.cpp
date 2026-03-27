#include "sdfg/data_flow/library_nodes/stdlib/memmove.h"

namespace sdfg {
namespace stdlib {

MemmoveNode::MemmoveNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::Expression count
)
    : StdlibNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Memmove,
          {"_dst"},
          {"_src"},
          true,
          data_flow::ImplementationType_NONE
      ),
      count_(count) {}

const symbolic::Expression MemmoveNode::count() const { return count_; }

void MemmoveNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet MemmoveNode::symbols() const {
    auto count_symbols = symbolic::atoms(this->count_);
    return count_symbols;
}

std::unique_ptr<data_flow::DataFlowNode> MemmoveNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<MemmoveNode>(element_id, debug_info_, vertex, parent, count_);
}

void MemmoveNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->count_ = symbolic::subs(this->count_, old_expression, new_expression);
}

nlohmann::json MemmoveNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MemmoveNode& node = static_cast<const MemmoveNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["count"] = serializer.expression(node.count());

    return j;
}

data_flow::LibraryNode& MemmoveNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("count"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Memmove.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Extract properties
    auto count = symbolic::parse(j.at("count"));

    return builder.add_library_node<MemmoveNode>(parent, debug_info, count);
}

MemmoveNodeDispatcher::MemmoveNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const MemmoveNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemmoveNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const MemmoveNode&>(node_);

    stream << language_extension_.external_prefix() << "memmove(" << node.outputs().at(0) << ", " << node.inputs().at(0)
           << ", " << language_extension_.expression(node.count()) << ")" << ";";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
