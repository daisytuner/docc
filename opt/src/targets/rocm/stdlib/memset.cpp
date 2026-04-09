#include "sdfg/targets/rocm/stdlib/memset.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg::rocm::stdlib {

MemsetNodeDispatcher_ROCMWithTransfers::MemsetNodeDispatcher_ROCMWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::stdlib::MemsetNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemsetNodeDispatcher_ROCMWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const sdfg::stdlib::MemsetNode&>(node_);

    globals_stream << "#include <hip/hip_runtime.h>" << std::endl;

    stream << "hipError_t err_hip;" << std::endl;

    std::string num_expr = language_extension_.expression(node.num());

    stream << "void *d_ptr;" << std::endl;
    stream << "err_hip = hipMalloc(&d_ptr, " << num_expr << ");" << std::endl;
    rocm_error_checking(stream, language_extension_, "err_hip");

    stream << "err_hip = hipMemset(d_ptr, " << language_extension_.expression(node.value()) << ", " << num_expr << ");"
           << std::endl;
    rocm_error_checking(stream, language_extension_, "err_hip");

    stream << "err_hip = hipMemcpy(" << node.outputs().at(0) << ", d_ptr, " << num_expr << ", hipMemcpyDeviceToHost);"
           << std::endl;
    rocm_error_checking(stream, language_extension_, "err_hip");

    stream << "err_hip = hipFree(d_ptr);" << std::endl;
    rocm_error_checking(stream, language_extension_, "err_hip");
}

MemsetNodeDispatcher_ROCMWithoutTransfers::MemsetNodeDispatcher_ROCMWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::stdlib::MemsetNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemsetNodeDispatcher_ROCMWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const sdfg::stdlib::MemsetNode&>(node_);

    globals_stream << "#include <hip/hip_runtime.h>" << std::endl;

    stream << "hipError_t err_hip;" << std::endl;
    stream << "err_hip = hipMemset(" << node.outputs().at(0) << ", " << language_extension_.expression(node.value())
           << ", " << language_extension_.expression(node.num()) << ");" << std::endl;
    rocm_error_checking(stream, language_extension_, "err_hip");
}

} // namespace sdfg::rocm::stdlib
