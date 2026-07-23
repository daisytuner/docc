/**
 * @file fft_conv_node.h
 * @brief Fused depthwise-convolution-via-FFT library node (hand-tuned GPU path).
 *
 * Unlike @ref FFTNode / @ref IFFTNode (which lower to cuFFT/hipFFT library calls and
 * a primitive complex-multiply map), FFTConvNode is a single fused operation whose
 * GPU dispatcher emits hardcoded mixed-radix Stockham FFT kernels (mirroring
 * `/home/adrian/fft_conv_tuned.cu`), operating on native complex (`CFloat`/`CDouble`)
 * buffers. It represents an entire 2D depthwise convolution:
 *
 *   Y[n,c] = crop( IFFT2( FFT2(pad(X[n,c])) * FFT2(pad(flip(W[c]))) ) ) [+ bias[c]]
 *
 * The node has no CPU expansion; it is realized only by the CUDA/ROCm hand-tuned
 * dispatchers selected via its implementation type.
 */

#pragma once

#include <vector>

#include "sdfg/data_flow/library_nodes/math/math_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_FFTConv("ml::FFTConv");

/**
 * @class FFTConvNode
 * @brief Fused 2D depthwise convolution executed via a hand-tuned FFT pipeline.
 */
class FFTConvNode : public MathNode {
private:
    std::vector<symbolic::Expression> shape_; ///< Input shape [N, C, H, W].
    std::vector<symbolic::Expression> kernel_shape_; ///< Kernel spatial extents [Kh, Kw].
    std::vector<symbolic::Expression> pads_; ///< Padding [top, left, bottom, right].
    types::PrimitiveType precision_; ///< Float or Double.
    bool with_bias_;

public:
    static constexpr int Y_INPUT_IDX = 0;
    static constexpr int X_INPUT_IDX = 1;
    static constexpr int W_INPUT_IDX = 2;
    static constexpr int B_INPUT_IDX = 3;

    FFTConvNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<symbolic::Expression>& kernel_shape,
        const std::vector<symbolic::Expression>& pads,
        types::PrimitiveType precision,
        bool with_bias
    );

    const std::vector<symbolic::Expression>& shape() const;
    const std::vector<symbolic::Expression>& kernel_shape() const;
    const std::vector<symbolic::Expression>& pads() const;
    types::PrimitiveType real_primitive() const;
    types::PrimitiveType complex_primitive() const;
    bool with_bias() const;

    void validate(const Function& function) const override;

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class FFTConvNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
