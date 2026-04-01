#pragma once

#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/serializer/json_serializer.h>

namespace sdfg {
namespace tenstorrent {

inline data_flow::LibraryNodeCode LibraryNodeType_Tenstorrent_CreateDevice("TTCreateDevice");

class TTCreateDevice : public data_flow::LibraryNode {
    friend class sdfg::builder::SDFGBuilder;
    friend class sdfg::builder::StructuredSDFGBuilder;

private:
    symbolic::Expression device_id_;

public:
    TTCreateDevice(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<std::string>& outputs,
        symbolic::Expression device_id
    );

    void validate(const Function& function) const override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    const symbolic::Expression device_id() const;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
};

class TTCreateDeviceDispatcher : public codegen::LibraryNodeDispatcher {
public:
    TTCreateDeviceDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

class TTCreateDeviceSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tenstorrent
} // namespace sdfg
