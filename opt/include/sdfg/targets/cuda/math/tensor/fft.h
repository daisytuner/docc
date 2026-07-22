#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"

namespace sdfg::cuda::tensor {

/**
 * @brief Emit a cuFFT error-checking guard for a `cufftResult` status variable.
 */
void cufft_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
);

/**
 * @brief Generate the cuFFT plan creation + execution for an FFT/IFFT node.
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

class FFTNodeDispatcher_CUFFTWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    FFTNodeDispatcher_CUFFTWithTransfers(
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

class FFTNodeDispatcher_CUFFTWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    FFTNodeDispatcher_CUFFTWithoutTransfers(
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

} // namespace sdfg::cuda::tensor
