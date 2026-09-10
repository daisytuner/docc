#include "sdfg/targets/cuda/tiles/async_copy_node.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/codegen/language_extensions/rocm_language_extension.h"

namespace sdfg {
namespace cuda {
namespace tiles {

CpAsyncCopyNodeDispatcher::CpAsyncCopyNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::CpAsyncCopyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void CpAsyncCopyNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    const auto& node = static_cast<const ::sdfg::tiles::CpAsyncCopyNode&>(node_);
    // Connector order {"_dst", "_src"} — both are addresses (reference memlets).
    const std::string& dst = inputs.at(0).expr;
    const std::string& src = inputs.at(1).expr;
    const size_t bytes = node.bytes();

    const size_t words = bytes / 4;
    out.stream << "__pipeline_memcpy_async(" << dst << ", " << src << ", " << bytes << ");" << std::endl;
}

VectorCopyNodeDispatcher::VectorCopyNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::VectorCopyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void VectorCopyNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    const auto& node = static_cast<const ::sdfg::tiles::VectorCopyNode&>(node_);
    // Connector order {"_dst", "_src"} — both are addresses (reference memlets).
    const std::string& dst = inputs.at(0).expr;
    const std::string& src = inputs.at(1).expr;
    // A width-matched integer vector: a raw byte move (no fp semantics), legal for
    // any element type as long as bytes is 4/8/16. int4/int2/int are builtin vector
    // types on both CUDA (vector_types.h) and HIP (hip_vector_types.h).
    const char* vec = node.bytes() == 16 ? "int4" : node.bytes() == 8 ? "int2" : "int";
    out.stream << "*reinterpret_cast<" << vec << "*>(" << dst << ") = *reinterpret_cast<const " << vec << "*>(" << src
               << ");" << std::endl;
}

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
