#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_conv_node.h"

namespace sdfg::cuda::tensor {

/**
 * @brief Hand-tuned CUDA dispatcher for the fused FFT depthwise-convolution node.
 *
 * Emits hardcoded mixed-radix Stockham FFT kernels (operating on native complex
 * `float2` buffers) into a `.cu` snippet and launches the full pipeline
 * (pad -> forward FFT -> complex multiply -> inverse FFT -> crop + bias),
 * managing its own device buffers and host<->device transfers -- mirroring the
 * hardcoded-kernel pattern used by the softmax dispatcher.
 *
 * v1 supports single-precision (float) depthwise convolutions.
 */
class FFTConvNodeDispatcher_CUDAWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    FFTConvNodeDispatcher_CUDAWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::FFTConvNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

/**
 * @brief Device-resident variant of @ref FFTConvNodeDispatcher_CUDAWithTransfers.
 *
 * Selected after @c CUDAFFTConvDataTransferExtraction has pulled the host<->device
 * copies out into explicit offloading nodes: the operands (X, W, bias, Y) are already
 * device pointers and are used directly. Only the internal FFT scratch buffers are
 * managed here.
 */
class FFTConvNodeDispatcher_CUDAWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    FFTConvNodeDispatcher_CUDAWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::FFTConvNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace sdfg::cuda::tensor
