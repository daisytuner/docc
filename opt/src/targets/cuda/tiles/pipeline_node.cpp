#include "sdfg/targets/cuda/tiles/pipeline_node.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/codegen/language_extensions/rocm_language_extension.h"

namespace sdfg {
namespace cuda {
namespace tiles {

PipelineCommitNodeDispatcher::PipelineCommitNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::PipelineCommitNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void PipelineCommitNodeDispatcher::
    dispatch(codegen::PrettyPrinter& stream, codegen::PrettyPrinter&, codegen::CodeSnippetFactory&) {
    stream << "__pipeline_commit();" << std::endl;
}

PipelineWaitNodeDispatcher::PipelineWaitNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::PipelineWaitNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void PipelineWaitNodeDispatcher::
    dispatch(codegen::PrettyPrinter& stream, codegen::PrettyPrinter&, codegen::CodeSnippetFactory&) {
    const auto& node = static_cast<const ::sdfg::tiles::PipelineWaitNode&>(node_);
    stream << "__pipeline_wait_prior(" << node.keep_outstanding() << ");" << std::endl;
}

} // namespace tiles
} // namespace cuda
} // namespace sdfg
