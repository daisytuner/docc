#include "sdfg/targets/cuda/math/tensor/conv_expander.h"

#include "sdfg/expanders/conv_expander.h"
#include "sdfg/expanders/conv_fft_expander.h"

namespace sdfg {
namespace offloading {

bool CudaConvExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Decision only: pick an expansion strategy. The actual lowering logic lives in the
    // target-neutral `expanders` layer.
    //   1) Frequency-domain FFT path (opt-in via DOCC_CONV_FFT, self-gated on applicability).
    //   2) im2row + GEMM (requires group == 1).
    //   3) Naïve direct convolution (fallback, any grouping).
    if (expanders::ConvFFTExpander::expand_conv_fft(builder, analysis_manager, node_)) {
        return true;
    }
    if (expanders::ConvExpander::expand_conv_im2row(builder, analysis_manager, node_)) {
        return true;
    }
    return expanders::ConvExpander::expand_conv_naive(builder, analysis_manager, node_);
}

} // namespace offloading
} // namespace sdfg
