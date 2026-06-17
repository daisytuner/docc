#pragma once

#include "sdfg/data_flow/library_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace data_flow {

inline LibraryNodeCode LibraryNodeType_Metadata("Metadata");

class MetadataNode : public LibraryNode {
private:
    std::unordered_map<std::string, std::string> metadata_;

public:
    MetadataNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        std::unordered_map<std::string, std::string> metadata
    );

    const std::unordered_map<std::string, std::string>& metadata() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

class MetadataNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const LibraryNode& library_node) override;

    LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class MetadataDispatcher : public codegen::LibraryNodeDispatcher {
public:
    MetadataDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const DataFlowGraph& data_flow_graph,
        const MetadataNode& node
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace data_flow
} // namespace sdfg
