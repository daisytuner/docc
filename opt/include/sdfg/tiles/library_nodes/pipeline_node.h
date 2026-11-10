#pragma once

#include <vector>

#include "sdfg/data_flow/library_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace tiles {

inline data_flow::LibraryNodeCode LibraryNodeType_PipelineCommit{"pipeline_commit"};
inline data_flow::LibraryNodeCode LibraryNodeType_PipelineWait{"pipeline_wait"};
inline data_flow::LibraryNodeCode LibraryNodeType_PipelineSchedule{"pipeline_schedule"};

/// Instruction classes a @ref PipelineScheduleNode orders.
enum class InstructionClass {
    Mfma, ///< matrix-core multiply-accumulate
    GlobalRead, ///< global memory read
    LdsRead, ///< shared / local data share read
    LdsWrite, ///< shared / local data share write
};

/// One step of an instruction schedule: the next @ref count instructions of class @ref cls.
struct ScheduleGroup {
    InstructionClass cls;
    size_t count;

    bool operator==(const ScheduleGroup& other) const {
        return cls == other.cls && count == other.count;
    }
};

class PipelineCommitNode : public data_flow::LibraryNode {
public:
    PipelineCommitNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type
    );

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

class PipelineWaitNode : public data_flow::LibraryNode {
    size_t keep_outstanding_;
    size_t loads_per_group_;

public:
    PipelineWaitNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        size_t keep_outstanding,
        size_t loads_per_group = 1
    );

    size_t keep_outstanding() const {
        return keep_outstanding_;
    }

    size_t loads_per_group() const {
        return loads_per_group_;
    }

    void set_loads_per_group(size_t loads_per_group) {
        loads_per_group_ = loads_per_group;
    }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

/**
 * @brief An instruction schedule for the code preceding it in its scheduling region.
 *
 * The groups are an ordered recipe for the backend instruction scheduler: issue the next
 * `count` instructions of class `cls`, then those of the next group, and so on (e.g. one
 * MFMA, one global read, one MFMA, ...). Instructions no group claims are scheduled freely,
 * so a recipe whose counts do not match the region exactly only weakens the schedule; it
 * never changes the program's semantics. Targets without such a scheduler hint ignore it.
 */
class PipelineScheduleNode : public data_flow::LibraryNode {
    std::vector<ScheduleGroup> groups_;

public:
    PipelineScheduleNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        std::vector<ScheduleGroup> groups
    );

    const std::vector<ScheduleGroup>& groups() const {
        return groups_;
    }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

/// Dispatcher for targets without an instruction-scheduler hint: emits nothing.
class PipelineScheduleNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    PipelineScheduleNodeDispatcher(
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

// ---- Serializers ---------------------------------------------------------

class PipelineCommitNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

class PipelineWaitNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

class PipelineScheduleNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

} // namespace tiles
} // namespace sdfg
