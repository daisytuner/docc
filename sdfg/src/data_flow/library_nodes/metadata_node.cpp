#include "sdfg/data_flow/library_nodes/metadata_node.h"

namespace sdfg {
namespace data_flow {

MetadataNode::MetadataNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    std::unordered_map<std::string, std::string> metadata
)
    : LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Metadata, outputs, inputs, false, ImplementationType_NONE
      ),
      metadata_(metadata) {}

void MetadataNode::validate(const Function& function) const {
    LibraryNode::validate(function);
    // No specific validation for metadata
}

const std::unordered_map<std::string, std::string>& MetadataNode::metadata() const { return metadata_; }

symbolic::SymbolSet MetadataNode::symbols() const { return symbolic::SymbolSet(); }

std::unique_ptr<DataFlowNode> MetadataNode::clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
    const {
    return std::make_unique<MetadataNode>(element_id, debug_info_, vertex, parent, outputs_, inputs_, metadata_);
}

void MetadataNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    // Do nothing
    return;
}

void MetadataNode::replace(const symbolic::ExpressionMapping& replacements) {
    // Do nothing
    return;
}

nlohmann::json MetadataNodeSerializer::serialize(const LibraryNode& library_node) {
    const MetadataNode& metadata_node = static_cast<const MetadataNode&>(library_node);
    nlohmann::json j;

    j["code"] = metadata_node.code().value();
    j["outputs"] = metadata_node.outputs();
    j["inputs"] = metadata_node.inputs();
    j["side_effect"] = metadata_node.side_effect();
    j["metadata"] = metadata_node.metadata();

    return j;
}

LibraryNode& MetadataNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("outputs"));
    assert(j.contains("inputs"));
    assert(j.contains("debug_info"));
    assert(j.contains("metadata"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Metadata.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Extract properties
    auto outputs = j.at("outputs").get<std::vector<std::string>>();
    auto inputs = j.at("inputs").get<std::vector<std::string>>();
    auto metadata = j.at("metadata").get<std::unordered_map<std::string, std::string>>();

    return builder.add_library_node<MetadataNode>(parent, debug_info, outputs, inputs, metadata);
}

MetadataDispatcher::MetadataDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const DataFlowGraph& data_flow_graph,
    const MetadataNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MetadataDispatcher::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    throw std::runtime_error("MetadataNode is not supported");
}

} // namespace data_flow
} // namespace sdfg
