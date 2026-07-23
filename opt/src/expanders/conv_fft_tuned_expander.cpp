#include "sdfg/expanders/conv_fft_tuned_expander.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_conv_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "symengine/integer.h"

namespace sdfg {
namespace expanders {

namespace {

bool is_const_int(const symbolic::Expression& e) { return SymEngine::is_a<SymEngine::Integer>(*e); }

bool all_const(const std::vector<symbolic::Expression>& v) {
    for (const auto& e : v) {
        if (!is_const_int(e)) {
            return false;
        }
    }
    return true;
}

} // namespace

bool ConvFFTTunedExpander::is_applicable(const math::tensor::ConvNode& node) {
    // 2D only.
    if (node.kernel_shape().size() != 2) {
        return false;
    }
    // Depthwise: one filter per input channel, one output channel per input channel.
    const auto& channels = node.shape()[1];
    if (!symbolic::eq(node.group(), channels) || !symbolic::eq(node.output_channels(), channels)) {
        return false;
    }
    // Unit strides and dilations.
    for (const auto& s : node.strides()) {
        if (!symbolic::eq(s, symbolic::one())) {
            return false;
        }
    }
    for (const auto& d : node.dilations()) {
        if (!symbolic::eq(d, symbolic::one())) {
            return false;
        }
    }
    // The hand-tuned dispatcher computes FFT sizes/radices at codegen time -> geometry must be constant.
    if (node.shape().size() != 4 || !all_const(node.shape()) || !all_const(node.kernel_shape()) ||
        !all_const(node.pads())) {
        return false;
    }
    return true;
}

bool ConvFFTTunedExpander::enabled() {
    auto env = getenv("DOCC_CONV_FFT_TUNED");
    if (env == nullptr) {
        return false;
    }
    std::string env_str(env);
    std::transform(env_str.begin(), env_str.end(), env_str.begin(), ::tolower);
    return env_str == "1" || env_str == "true";
}

bool ConvFFTTunedExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return expand_conv_fft_tuned(builder, analysis_manager, node_);
}

bool ConvFFTTunedExpander::expand_conv_fft_tuned(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, math::tensor::ConvNode& node
) {
    if (!enabled() || !is_applicable(node)) {
        return false;
    }

    auto& dfg = node.get_parent();
    math::tensor::ConvNode::ConvExpandPrerequisits b;
    if (!node.check_expandable(dfg, b)) {
        return false;
    }

    // v1: single precision only.
    if (node.primitive_type(dfg) != types::PrimitiveType::Float) {
        return false;
    }

    const DebugInfo& dbg = node.debug_info();

    // New sequence that replaces the convolution block.
    auto& new_sequence = builder.add_sequence_before(
        *b.block_parent, *b.block, b.block_parent->at(b.block_index).second.assignments(), b.block->debug_info()
    );
    auto& blk = builder.add_block(new_sequence, {}, b.block->debug_info());

    auto& fftconv = builder.add_library_node<math::tensor::FFTConvNode>(
        blk,
        dbg,
        cuda::ImplementationType_CUDAWithTransfers,
        node.shape(),
        node.kernel_shape(),
        node.pads(),
        types::PrimitiveType::Float,
        b.has_bias
    );

    auto& y_acc = builder.add_access(blk, b.access_Y->data(), b.access_Y->debug_info());
    builder.add_computational_memlet(blk, y_acc, fftconv, "Y", {}, b.iedge_Y->base_type(), b.iedge_Y->debug_info());
    auto& x_acc = builder.add_access(blk, b.access_X->data(), b.access_X->debug_info());
    builder.add_computational_memlet(blk, x_acc, fftconv, "X", {}, b.iedge_X->base_type(), b.iedge_X->debug_info());
    auto& w_acc = builder.add_access(blk, b.access_W->data(), b.access_W->debug_info());
    builder.add_computational_memlet(blk, w_acc, fftconv, "W", {}, b.iedge_W->base_type(), b.iedge_W->debug_info());
    if (b.has_bias) {
        auto& b_acc = builder.add_access(blk, b.access_B->data(), b.access_B->debug_info());
        builder.add_computational_memlet(blk, b_acc, fftconv, "B", {}, b.iedge_B->base_type(), b.iedge_B->debug_info());
    }

    // Remove the original convolution block.
    builder.clear_code_node_legacy(*b.block, node);
    builder.remove_child(*b.block_parent, b.block_index + 1);

    return true;
}

} // namespace expanders
} // namespace sdfg
