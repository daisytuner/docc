#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/r2c_fft2d_node.h"

namespace sdfg::cuda::tensor {

/**
 * @brief Hand-tuned CUDA dispatcher for the forward 2D real-to-complex FFT node.
 *
 * Emits mixed-radix Stockham kernels (R2C rows + column FFT) that transform a real,
 * already-padded input [matrices, fftH, fftW] into the Hermitian half spectrum
 * [matrices, fftH, halfW], managing its own device buffers and host<->device
 * transfers.
 */
class R2CFFT2DNodeDispatcher_CUDAWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    R2CFFT2DNodeDispatcher_CUDAWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::R2CFFT2DNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

/**
 * @brief Device-resident variant of @ref R2CFFT2DNodeDispatcher_CUDAWithTransfers.
 *
 * The operands (Y, X) are already device pointers; only the internal FFT scratch
 * (radix tables) is managed here.
 */
class R2CFFT2DNodeDispatcher_CUDAWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    R2CFFT2DNodeDispatcher_CUDAWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::R2CFFT2DNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace sdfg::cuda::tensor
