#pragma once

#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/stdlib_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace stdlib {

inline data_flow::LibraryNodeCode LibraryNodeType_Malloc("Malloc");

class MallocNode : public StdlibNode {
private:
    symbolic::Expression size_;

public:
    MallocNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const symbolic::Expression size
    );

    const symbolic::Expression size() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    std::string toStr() const override;
};

class MallocNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

class MallocNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    MallocNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const MallocNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace stdlib
} // namespace sdfg
