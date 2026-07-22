#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

namespace sdfg {
namespace expanders {

/**
 * @class ConvFFTExpander
 * @brief Expands a 2D depthwise convolution into an FFT -> complexMul -> inverse-FFT pipeline.
 *
 * This is the frequency-domain equivalent of a spatial depthwise convolution
 * (mirroring `cufft_conv.cu`). The convolution theorem turns the spatial
 * cross-correlation into a pointwise complex multiplication in frequency space:
 *
 *   Y = crop( IFFT( FFT(pad(X)) * FFT(pad(flip(W))) ) ) / (padH*padW) [+ bias]
 *
 * The produced `FFTNode` / `IFFTNode` library nodes are created with
 * `ImplementationType_NONE`; the per-target offloading rewriter later selects the
 * cuFFT / hipFFT implementation, so this expander is target-neutral and lives in
 * the shared `expanders` layer (used by both the CUDA and ROCm conv expanders).
 *
 * ## Preconditions (otherwise `expand` returns false and a fallback expander should be used)
 * - 2D convolution (`kernel_shape.size() == 2`).
 * - Depthwise: `group == C_in` and `output_channels == C_in`.
 * - Unit strides and unit dilations.
 * - Floating point (Float or Double) element type.
 */
class ConvFFTExpander {
private:
    math::tensor::ConvNode& node_;

public:
    explicit ConvFFTExpander(math::tensor::ConvNode& library_node) : node_(library_node) {}

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    /// @brief Whether @p node satisfies the preconditions above (cheap structural check).
    static bool is_applicable(const math::tensor::ConvNode& node);

    /// @brief Whether the FFT convolution path is enabled via the `DOCC_CONV_FFT` env var.
    /// Disabled by default so the spatial im2row / naive expansions remain the default lowering.
    static bool enabled();

    static bool expand_conv_fft(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        math::tensor::ConvNode& node
    );
};

} // namespace expanders
} // namespace sdfg
