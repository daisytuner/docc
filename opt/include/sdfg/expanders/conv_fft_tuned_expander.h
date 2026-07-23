#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

namespace sdfg {
namespace expanders {

/**
 * @class ConvFFTTunedExpander
 * @brief Lowers a 2D depthwise convolution into a single fused FFTConvNode that is
 *        realized by the hand-tuned mixed-radix Stockham FFT dispatcher.
 *
 * This is the alternative to @ref ConvFFTExpander: instead of emitting FFT/IFFT
 * library nodes + a primitive complex-multiply map (cuFFT path), it produces one
 * `FFTConvNode` whose GPU dispatcher emits hardcoded FFT kernels. Opt-in via the
 * `DOCC_CONV_FFT_TUNED` environment variable.
 *
 * ## Preconditions (else `expand_conv_fft_tuned` returns false; a fallback expander runs)
 * - 2D convolution with constant integer geometry (N, C, H, W, Kh, Kw, pads).
 * - Depthwise: `group == C_in` and `output_channels == C_in`.
 * - Unit strides and unit dilations.
 * - Single-precision (Float) element type.
 */
class ConvFFTTunedExpander {
private:
    math::tensor::ConvNode& node_;

public:
    explicit ConvFFTTunedExpander(math::tensor::ConvNode& library_node) : node_(library_node) {}

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static bool is_applicable(const math::tensor::ConvNode& node);

    /// @brief Whether the hand-tuned FFT conv path is enabled via `DOCC_CONV_FFT_TUNED`.
    static bool enabled();

    static bool expand_conv_fft_tuned(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        math::tensor::ConvNode& node
    );
};

} // namespace expanders
} // namespace sdfg
