#include "sdfg/targets/rocm/stdlib/memcpy.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg::rocm::stdlib {

MemcpyNodeDispatcher_ROCMWithTransfers::MemcpyNodeDispatcher_ROCMWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::stdlib::MemcpyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemcpyNodeDispatcher_ROCMWithTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const sdfg::stdlib::MemcpyNode&>(node_);

    out.library_snippet_factory.add_global("#include <hip/hip_runtime.h>");

    out.stream << "hipError_t err_hip;" << std::endl;

    std::string num_expr = language_extension_.expression(node.count());

    out.stream << "void *d_ptr_in;" << std::endl;
    out.stream << "void *d_ptr_out;" << std::endl;

    out.stream << "err_hip = hipMalloc(&d_ptr_in, " << num_expr << ");" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");
    out.stream << "err_hip = hipMalloc(&d_ptr_out, " << num_expr << ");" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");

    out.stream << "err_hip = hipMemcpy(d_ptr_in, " << inputs.at(1).expr << ", " << num_expr
               << ", hipMemcpyHostToDevice);" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");

    out.stream << "err_hip = hipMemcpy(d_ptr_out, d_ptr_in, " << num_expr << ", hipMemcpyDeviceToDevice);" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");

    out.stream << "err_hip = hipMemcpy(" << inputs.at(0).expr << ", d_ptr_out, " << num_expr
               << ", hipMemcpyDeviceToHost);" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");

    out.stream << "err_hip = hipFree(d_ptr_in);" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");
    out.stream << "err_hip = hipFree(d_ptr_out);" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");
}

MemcpyNodeDispatcher_ROCMWithoutTransfers::MemcpyNodeDispatcher_ROCMWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::stdlib::MemcpyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void MemcpyNodeDispatcher_ROCMWithoutTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const sdfg::stdlib::MemcpyNode&>(node_);

    out.library_snippet_factory.add_global("#include <hip/hip_runtime.h>");

    out.stream << "hipError_t err_hip;" << std::endl;
    out.stream << "err_hip = hipMemcpy(" << inputs.at(0).expr << ", " << inputs.at(1).expr << ", "
               << language_extension_.expression(node.count()) << ", hipMemcpyDeviceToDevice);" << std::endl;
    rocm_error_checking(out.stream, language_extension_, "err_hip");
}

} // namespace sdfg::rocm::stdlib
