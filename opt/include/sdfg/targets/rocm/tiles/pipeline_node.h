#pragma once

#include "sdfg/tiles/library_nodes/pipeline_node.h"

namespace sdfg {
namespace rocm {
namespace tiles {

class PipelineCommitNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    PipelineCommitNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const ::sdfg::tiles::PipelineCommitNode& node
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

class PipelineWaitNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    PipelineWaitNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const ::sdfg::tiles::PipelineWaitNode& node
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

/// Emits a @ref ::sdfg::tiles::PipelineScheduleNode as `__builtin_amdgcn_sched_group_barrier`s on CDNA.
class PipelineScheduleNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    PipelineScheduleNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const ::sdfg::tiles::PipelineScheduleNode& node
    );

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace tiles
} // namespace rocm
} // namespace sdfg
