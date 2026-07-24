#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/tensor/c2r_fft2d_node.h"

namespace sdfg::cuda::tensor {

/**
 * @brief Hand-tuned CUDA dispatcher for the inverse 2D complex-to-real FFT node.
 *
 * Emits mixed-radix Stockham kernels (column FFT + C2R rows) that transform the
 * Hermitian half spectrum [matrices, fftH, halfW] back to a real signal
 * [matrices, fftH, fftW], scaled by 1/(fftH*fftW), managing its own device buffers
 * and host<->device transfers.
 */
class C2RFFT2DNodeDispatcher_CUDAWithTransfers : public codegen::LibraryNodeDispatcher {
public:
    C2RFFT2DNodeDispatcher_CUDAWithTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::C2RFFT2DNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

/**
 * @brief Device-resident variant of @ref C2RFFT2DNodeDispatcher_CUDAWithTransfers.
 *
 * The operands (Y, X) are already device pointers; only the internal FFT scratch
 * (radix tables) is managed here.
 */
class C2RFFT2DNodeDispatcher_CUDAWithoutTransfers : public codegen::LibraryNodeDispatcher {
public:
    C2RFFT2DNodeDispatcher_CUDAWithoutTransfers(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::C2RFFT2DNode& node
    );

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

} // namespace sdfg::cuda::tensor
