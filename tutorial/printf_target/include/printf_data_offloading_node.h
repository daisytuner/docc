#pragma once

#include <cstddef>

#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/graph/graph.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/targets/offloading/data_offloading_node.h>

namespace sdfg {
namespace printf_target {

/// Library node type identifier for printf offloading operations
inline data_flow::LibraryNodeCode LibraryNodeType_Printf_Offloading("PrintfOffloading");

/**
 * @brief Data offloading node that generates printf statements instead of actual transfers
 *
 * This node replaces CUDA-style memory operations (malloc, memcpy, free) with
 * printf statements that trace what would happen, useful for debugging and
 * understanding data flow without actual device execution.
 */
class PrintfDataOffloadingNode : public offloading::DataOffloadingNode {
public:
    PrintfDataOffloadingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        symbolic::Expression size,
        offloading::DataTransferDirection transfer_direction,
        offloading::BufferLifecycle buffer_lifecycle
    );

    /// Validates the node configuration
    void validate(const Function& function) const override;

    /// Creates a deep copy of this node
    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    /// Returns all symbols used by this node
    symbolic::SymbolSet symbols() const override;

    /// Replaces occurrences of old_expression with new_expression
    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    /// Printf operations are non-blocking (just print and continue)
    virtual bool blocking() const override;

    /// Check if this node is redundant with another offloading node
    virtual bool redundant_with(const offloading::DataOffloadingNode& other) const override;

    /// Check if this node is equal to another offloading node
    virtual bool equal_with(const offloading::DataOffloadingNode& other) const override;

    virtual bool is_same_target(const DataOffloadingNode& other) const override;
};

/**
 * @brief Dispatcher that generates printf code for data offloading operations
 */
class PrintfDataOffloadingNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    PrintfDataOffloadingNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    /// Generates printf statements for the data transfer operation
    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    /// Returns instrumentation info for this node
    virtual codegen::InstrumentationInfo instrumentation_info() const override;
};

/**
 * @brief Serializer for saving/loading PrintfDataOffloadingNode to/from JSON
 */
class PrintfDataOffloadingNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    /// Serializes the node to JSON
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    /// Deserializes a node from JSON
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

} // namespace printf_target
} // namespace sdfg
