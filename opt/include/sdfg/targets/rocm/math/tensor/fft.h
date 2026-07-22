#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"

namespace sdfg::rocm::tensor {

/**
 * @brief Emit a hipFFT error-checking guard for a `hipfftResult` status variable.
 */
void hipfft_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
);

/**
 * @brief Generate the hipFFT plan creation + execution for an FFT/IFFT node.
 *
 * Handles both forward (R2C / D2Z) and inverse (C2R / Z2D) transforms based on
 * `node.direction()`, using the Hermitian layout for the complex side. When
 * `with_transfers` is true, host<->device buffers are allocated and copied;
 * otherwise `__X` / `__Y` are assumed to already reside on the device.
 */
void generate_fft(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const math::tensor::FFTNodeBase& node,
    bool with_transfers
);

class FFTNodeDispatcher_HIPFFTWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    FFTNodeDispatcher_HIPFFTWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::FFTNodeBase& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

class FFTNodeDispatcher_HIPFFTWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    FFTNodeDispatcher_HIPFFTWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::FFTNodeBase& node
    );

    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace sdfg::rocm::tensor
