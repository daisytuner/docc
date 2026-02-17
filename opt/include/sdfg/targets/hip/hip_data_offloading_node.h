#pragma once

#include <cstddef>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg {
namespace hip {

inline data_flow::LibraryNodeCode LibraryNodeType_HIP_Offloading("HIPOffloading");

class HIPDataOffloadingNode : public offloading::DataOffloadingNode {
private:
    symbolic::Expression device_id_;

public:
    HIPDataOffloadingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        symbolic::Expression size,
        symbolic::Expression device_id,
        offloading::DataTransferDirection transfer_direction,
        offloading::BufferLifecycle buffer_lifecycle
    );

    void validate(const Function& function) const override;

    const symbolic::Expression device_id() const;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    virtual bool blocking() const override;

    virtual bool redundant_with(const offloading::DataOffloadingNode& other) const override;

    virtual bool equal_with(const offloading::DataOffloadingNode& other) const override;
};

class HIPDataOffloadingNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    HIPDataOffloadingNodeDispatcher(
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

    virtual codegen::InstrumentationInfo instrumentation_info() const override;
};

class HIPDataOffloadingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

} // namespace hip
} // namespace sdfg
