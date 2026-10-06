#include "sdfg/targets/gpu/barrier_local_node_dispatcher.h"

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/targets/cuda/codegen/cuda_language_extension.h"
#include "sdfg/targets/rocm/codegen/rocm_language_extension.h"

namespace sdfg::gpu {

BarrierLocalNodeDispatcher::BarrierLocalNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::BarrierLocalNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void BarrierLocalNodeDispatcher::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    if (dynamic_cast<codegen::CLanguageExtension*>(&this->language_extension_) != nullptr) {
        throw std::runtime_error(
            "ThreadBarrierDispatcher is not supported for C language extension. Use CUDA or ROCM "
            "language extension instead."
        );
    } else if (dynamic_cast<codegen::CPPLanguageExtension*>(&this->language_extension_) != nullptr) {
        throw std::runtime_error(
            "ThreadBarrierDispatcher is not supported for C++ language extension. Use CUDA or ROCM "
            "language extension instead."
        );
    } else if (dynamic_cast<::sdfg::cuda::CUDALanguageExtension*>(&this->language_extension_) != nullptr) {
        stream << "__syncthreads();" << std::endl;
    } else if (dynamic_cast<::sdfg::rocm::ROCMLanguageExtension*>(&this->language_extension_) != nullptr) {
        stream << "__syncthreads();" << std::endl;
    } else {
        throw std::runtime_error("Unsupported language extension for ThreadBarrierDispatcher");
    }
}

} // namespace sdfg::gpu
