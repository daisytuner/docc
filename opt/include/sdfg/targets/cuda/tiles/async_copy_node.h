#pragma once

#include "sdfg/tiles/library_nodes/async_copy_node.h"

namespace sdfg {
namespace cuda {
namespace tiles {

class CpAsyncCopyNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    CpAsyncCopyNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const ::sdfg::tiles::CpAsyncCopyNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

class VectorCopyNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    VectorCopyNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const ::sdfg::tiles::VectorCopyNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

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

} // namespace tiles
} // namespace cuda
} // namespace sdfg
