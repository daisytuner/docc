#pragma once

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg {
namespace offloading {

inline data_flow::LibraryNodeCode LibraryNodeType_External_Offloading("ExternalOffloading");

class ExternalDataOffloadingNode : public offloading::DataOffloadingNode {
private:
    std::string callee_name_;
    size_t transfer_index_;

public:
    ExternalDataOffloadingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<std::string>& inputs,
        const std::string& callee_name,
        size_t transfer_index,
        DataTransferDirection transfer_direction,
        BufferLifecycle buffer_lifecycle
    );

    const std::string& callee_name() const;

    size_t transfer_index() const;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    void validate(const Function& function) const override;

    virtual bool blocking() const override;

    virtual bool redundant_with(const offloading::DataOffloadingNode& other) const override;

    virtual bool equal_with(const offloading::DataOffloadingNode& other) const override;

    virtual bool is_same_target(const offloading::DataOffloadingNode& other) const override;
};

class ExternalDataOffloadingNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    ExternalDataOffloadingNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

class ExternalDataOffloadingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace offloading
} // namespace sdfg
