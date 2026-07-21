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

void MemsetNodeDispatcher_ROCMWithTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const sdfg::stdlib::MemsetNode&>(node_);

    out.library_snippet_factory.add_global("#include <hip/hip_runtime.h>");

    out.stream << "hipError_t err_hip;" << std::endl;

    std::string num_expr = language_extension_.expression(node.num());

    out.stream << "void *d_ptr;" << std::endl;
    out.stream << "err_hip = hipMalloc(&d_ptr, " << num_expr << ");" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");

    out.stream << "err_hip = hipMemset(d_ptr, " << language_extension_.expression(node.value()) << ", " << num_expr
               << ");" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");

    out.stream << "err_hip = hipMemcpy(" << inputs.at(0).expr << ", d_ptr, " << num_expr << ", hipMemcpyDeviceToHost);"
               << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");

    out.stream << "err_hip = hipFree(d_ptr);" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");
}

MemsetNodeDispatcher_ROCMWithoutTransfers::MemsetNodeDispatcher_ROCMWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::stdlib::MemsetNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemsetNodeDispatcher_ROCMWithoutTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const sdfg::stdlib::MemsetNode&>(node_);

    out.library_snippet_factory.add_global("#include <hip/hip_runtime.h>");

    out.stream << "hipError_t err_hip;" << std::endl;
    out.stream << "err_hip = hipMemset(" << inputs.at(0).expr << ", " << language_extension_.expression(node.value())
               << ", " << language_extension_.expression(node.num()) << ");" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");
}

} // namespace sdfg::rocm::stdlib
