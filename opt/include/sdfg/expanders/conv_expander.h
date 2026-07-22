#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

namespace sdfg {
namespace expanders {

/**
 * @class ConvExpander
 * @brief Target-neutral spatial convolution expansion logic.
 *
 * Holds the actual lowering of a `ConvNode` into primitive SDFG constructs. It
 * is independent of any GPU target; the per-target expanders (CUDA / ROCm) own
 * only the *decision* of which expansion to attempt and in what order, and call
 * into these methods.
 *
 * - `expand_conv_im2row`: patch-extraction + GEMM lowering (requires `group == 1`).
 * - `expand_conv_naive`: direct nested-loop convolution (any grouping).
 *
 * The frequency-domain lowering lives separately in `ConvFFTExpander`.
 */
class ConvExpander {
public:
    /// @brief Naïve direct convolution as nested maps/loops. Works for any grouping.
    static bool expand_conv_naive(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        math::tensor::ConvNode& node
    );

    /// @brief im2row patch extraction + a single GEMM. Only valid when `group == 1`.
    static bool expand_conv_im2row(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        math::tensor::ConvNode& node
    );
};

} // namespace expanders
} // namespace sdfg
