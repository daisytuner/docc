#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"

namespace sdfg {
namespace stdlib {

MallocNode::MallocNode(
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
          LibraryNodeType_Malloc,
          {"_ret"},
          {},
          true, // debatable. Its a big change and we may want it as a flag
          data_flow::ImplementationType_NONE
      ),
      size_(size) {}

const symbolic::Expression MallocNode::size() const { return size_; }

void MallocNode::validate(const Function& function) const { LibraryNode::validate(function); }

symbolic::SymbolSet MallocNode::symbols() const { return symbolic::atoms(this->size_); }

std::unique_ptr<data_flow::DataFlowNode> MallocNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<MallocNode>(element_id, debug_info_, vertex, parent, size_);
}

void MallocNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->size_ = symbolic::subs(this->size_, old_expression, new_expression);
}

std::string MallocNode::toStr() const { return LibraryNode::toStr() + "(" + size_->__str__() + ")"; }

nlohmann::json MallocNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MallocNode& node = static_cast<const MallocNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();

    sdfg::serializer::JSONSerializer serializer;
    j["size"] = serializer.expression(node.size());

    return j;
}

data_flow::LibraryNode& MallocNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("size"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Malloc.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto size = symbolic::parse(j.at("size"));

    return builder.add_library_node<MallocNode>(parent, debug_info, size);
}

MallocNodeDispatcher::MallocNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const MallocNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MallocNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& malloc_node = static_cast<const MallocNode&>(node_);

    auto& graph = malloc_node.get_parent();
    auto& oedge = *graph.out_edges(malloc_node).begin();

    stream << malloc_node.outputs().at(0);
    stream << " = ";
    stream << "("
           << language_extension_.type_cast(
                  language_extension_.external_prefix() + "malloc(" +
                      language_extension_.expression(malloc_node.size()) + ")",
                  oedge.base_type()
              )
           << ");";
    stream << std::endl;
}

} // namespace stdlib
} // namespace sdfg
