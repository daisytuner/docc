#pragma once

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/stdlib_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace stdlib {

inline data_flow::LibraryNodeCode LibraryNodeType_Unreachable("Unreachable");

class UnreachableNode : public StdlibNode {
private:
    symbolic::Expression size_;

public:
    UnreachableNode(
        size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex, data_flow::DataFlowGraph& parent
    );

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

class UnreachableNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class UnreachableNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    UnreachableNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const UnreachableNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace stdlib
} // namespace sdfg
