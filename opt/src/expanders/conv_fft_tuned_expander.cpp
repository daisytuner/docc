#include "sdfg/expanders/conv_fft_tuned_expander.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/c2r_fft2d_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_conv_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/r2c_fft2d_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
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

// Compile-time integer value of a constant expression (guarded by all_const).
int64_t int_of(const symbolic::Expression& e) { return SymEngine::down_cast<const SymEngine::Integer&>(*e).as_int(); }

// Smallest 5-smooth number (factors 2,3,5 only) that is >= n. Matches the FFT dispatcher.
int64_t next_smooth(int64_t n) {
    for (int64_t x = std::max<int64_t>(n, 1);; ++x) {
        int64_t t = x;
        while (t % 2 == 0) t /= 2;
        while (t % 3 == 0) t /= 3;
        while (t % 5 == 0) t /= 5;
        if (t == 1) return x;
    }
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

    // Constant geometry (guaranteed const by is_applicable).
    const symbolic::Expression N = node.shape()[0];
    const symbolic::Expression C = node.shape()[1];
    const symbolic::Expression Hh = node.shape()[2];
    const symbolic::Expression Ww = node.shape()[3];
    const symbolic::Expression Kh = node.kernel_shape()[0];
    const symbolic::Expression Kw = node.kernel_shape()[1];
    const int64_t H_i = int_of(Hh), W_i = int_of(Ww), Kh_i = int_of(Kh), Kw_i = int_of(Kw);
    const int64_t ph_i = int_of(node.pads()[0]), pw_i = int_of(node.pads()[1]);
    const int64_t fftH_i = next_smooth(H_i + Kh_i - 1);
    const int64_t fftW_i = next_smooth(W_i + Kw_i - 1);
    const int64_t halfW_i = fftW_i / 2 + 1;
    const symbolic::Expression fftH = symbolic::integer(fftH_i);
    const symbolic::Expression fftW = symbolic::integer(fftW_i);
    const symbolic::Expression halfW = symbolic::integer(halfW_i);
    // Cross-correlation (torch Conv2d): flip in the spectral domain, both operands at the
    // origin -> the valid "same"-padding output window starts at (K - 1 - pad).
    const symbolic::Expression cropH = symbolic::integer(Kh_i - 1 - ph_i);
    const symbolic::Expression cropW = symbolic::integer(Kw_i - 1 - pw_i);

    // New sequence that replaces the convolution block.
    auto& new_sequence = builder.add_sequence_before(*b.block_parent, *b.block, b.block->debug_info());

    types::Scalar real_scalar(types::PrimitiveType::Float);
    types::Scalar cplx_scalar(types::PrimitiveType::CFloat);
    types::Pointer real_ptr(real_scalar);
    types::Pointer cplx_ptr(cplx_scalar);
    types::Scalar indvar_type(types::PrimitiveType::Int64);

    // Nested sequential maps over `extents`; the CUDA scheduler promotes them to kernels.
    auto add_maps = [&](structured_control_flow::Sequence& seq, const std::vector<symbolic::Expression>& extents
                    ) -> std::pair<structured_control_flow::Sequence*, std::vector<symbolic::Expression>> {
        structured_control_flow::Sequence* cur = &seq;
        std::vector<symbolic::Expression> ivs;
        for (const auto& ext : extents) {
            auto name = builder.find_new_name("_i");
            builder.add_container(name, indvar_type);
            auto iv = symbolic::symbol(name);
            auto& m = builder.add_map(
                *cur,
                iv,
                symbolic::Lt(iv, ext),
                symbolic::zero(),
                symbolic::add(iv, symbolic::one()),
                structured_control_flow::ScheduleType_Sequential::create(),
                dbg
            );
            cur = &m.root();
            ivs.push_back(iv);
        }
        return {cur, ivs};
    };

    // Row-major linearisation of a multi-index into a single flat offset. All operand
    // and transient buffers are flat pointers, so every memlet uses a 1-element subset.
    auto lin = [&](const std::vector<symbolic::Expression>& idx,
                   const std::vector<symbolic::Expression>& ext) -> data_flow::Subset {
        symbolic::Expression off = symbolic::zero();
        for (size_t d = 0; d < idx.size(); ++d) {
            off = symbolic::add(symbolic::mul(off, ext[d]), idx[d]);
        }
        return data_flow::Subset{off};
    };

    // Malloc'd transient buffer holding `count` elements of `elem`.
    auto make_buf = [&](const std::string& hint,
                        const types::Pointer& ptype,
                        const types::Scalar& elem,
                        const symbolic::Expression& count) -> std::string {
        auto name = builder.find_new_name(hint);
        builder.add_container(name, ptype);
        stdlib::add_malloc_block(
            builder, new_sequence, name, symbolic::mul(count, symbolic::size_of_type(elem)), ptype, dbg
        );
        return name;
    };

    const symbolic::Expression NC = symbolic::mul(N, C);
    const symbolic::Expression real_img_count = symbolic::mul(symbolic::mul(NC, fftH), fftW);
    const symbolic::Expression real_ker_count = symbolic::mul(symbolic::mul(C, fftH), fftW);
    const symbolic::Expression spec_img_count = symbolic::mul(symbolic::mul(NC, fftH), halfW);
    const symbolic::Expression spec_ker_count = symbolic::mul(symbolic::mul(C, fftH), halfW);

    const std::string padImg = make_buf("_fc_pad_img", real_ptr, real_scalar, real_img_count);
    const std::string padKer = make_buf("_fc_pad_ker", real_ptr, real_scalar, real_ker_count);
    const std::string specImg = make_buf("_fc_spec_img", cplx_ptr, cplx_scalar, spec_img_count);
    const std::string specKer = make_buf("_fc_spec_ker", cplx_ptr, cplx_scalar, spec_ker_count);
    const std::string invReal = make_buf("_fc_inv", real_ptr, real_scalar, real_img_count);

    // Zero-fill a real padded buffer over `extents`.
    auto zero_fill = [&](const std::string& buf, const std::vector<symbolic::Expression>& extents) {
        auto [seq, iv] = add_maps(new_sequence, extents);
        auto& cb = builder.add_block(*seq, {}, dbg);
        auto& zero = builder.add_constant(cb, "0.0", real_scalar, dbg);
        auto& acc = builder.add_access(cb, buf, dbg);
        auto& t = builder.add_tasklet(cb, data_flow::TaskletCode::assign, "_out", {"_in"}, dbg);
        builder.add_computational_memlet(cb, zero, t, "_in", {}, dbg);
        builder.add_computational_memlet(cb, t, "_out", acc, lin(iv, extents), real_ptr, dbg);
    };

    // --- Pad image: zero-fill [N,C,fftH,fftW] then copy X into the [H,W] origin window. ---
    zero_fill(padImg, {N, C, fftH, fftW});
    {
        auto [seq, iv] = add_maps(new_sequence, {N, C, Hh, Ww});
        auto& cb = builder.add_block(*seq, {}, dbg);
        auto& xacc = builder.add_access(cb, b.access_X->data(), dbg);
        auto& pacc = builder.add_access(cb, padImg, dbg);
        auto& t = builder.add_tasklet(cb, data_flow::TaskletCode::assign, "_out", {"_in"}, dbg);
        builder.add_computational_memlet(cb, xacc, t, "_in", lin(iv, {N, C, Hh, Ww}), real_ptr, dbg);
        builder.add_computational_memlet(cb, t, "_out", pacc, lin(iv, {N, C, fftH, fftW}), real_ptr, dbg);
    }

    // --- Pad kernel: zero-fill [C,fftH,fftW] then flip-copy W into the [Kh,Kw] origin window. ---
    zero_fill(padKer, {C, fftH, fftW});
    {
        auto [seq, iv] = add_maps(new_sequence, {C, Kh, Kw});
        auto& cb = builder.add_block(*seq, {}, dbg);
        auto& wacc = builder.add_access(cb, b.access_W->data(), dbg);
        auto& pacc = builder.add_access(cb, padKer, dbg);
        auto& t = builder.add_tasklet(cb, data_flow::TaskletCode::assign, "_out", {"_in"}, dbg);
        // Read W forward (positive strides so the offload range analysis succeeds) and
        // apply the cross-correlation flip on the write into the transient PadKer:
        // PadKer[c, Kh-1-r, Kw-1-w] = W[c, r, w].
        std::vector<symbolic::Expression> pad_idx = {
            iv[0],
            symbolic::sub(symbolic::sub(Kh, symbolic::one()), iv[1]),
            symbolic::sub(symbolic::sub(Kw, symbolic::one()), iv[2])
        };
        builder.add_computational_memlet(cb, wacc, t, "_in", lin(iv, {C, Kh, Kw}), real_ptr, dbg);
        builder.add_computational_memlet(cb, t, "_out", pacc, lin(pad_idx, {C, fftH, fftW}), real_ptr, dbg);
    }

    // --- Forward 2D FFTs (image and kernel) into the half spectra. ---
    auto add_r2c = [&](const std::string& out_spec, const std::string& in_pad, const symbolic::Expression& matrices) {
        auto& blk = builder.add_block(new_sequence, {}, dbg);
        auto& fnode = builder.add_library_node<math::tensor::R2CFFT2DNode>(
            blk,
            dbg,
            cuda::ImplementationType_CUDAWithTransfers,
            std::vector<symbolic::Expression>{matrices, fftH, fftW},
            types::PrimitiveType::Float
        );
        auto& yacc = builder.add_access(blk, out_spec, dbg);
        builder.add_computational_memlet(blk, yacc, fnode, "Y", {}, cplx_ptr, dbg);
        auto& xacc = builder.add_access(blk, in_pad, dbg);
        builder.add_computational_memlet(blk, xacc, fnode, "X", {}, real_ptr, dbg);
    };
    add_r2c(specImg, padImg, NC);
    add_r2c(specKer, padKer, C);

    // --- Pointwise complex multiply over the half spectrum (kernel broadcast over N). ---
    {
        auto [seq, iv] = add_maps(new_sequence, {N, C, fftH, halfW});
        auto& cb = builder.add_block(*seq, {}, dbg);
        auto& imgR = builder.add_access(cb, specImg, dbg);
        auto& kerR = builder.add_access(cb, specKer, dbg);
        auto& imgW = builder.add_access(cb, specImg, dbg);
        auto& t = builder.add_tasklet(cb, data_flow::TaskletCode::complex_mul, "_out", {"_in1", "_in2"}, dbg);
        builder.add_computational_memlet(cb, imgR, t, "_in1", lin(iv, {N, C, fftH, halfW}), cplx_ptr, dbg);
        // SpecKer indexed [c, h, w] (broadcast over batch n).
        std::vector<symbolic::Expression> ker_idx = {iv[1], iv[2], iv[3]};
        builder.add_computational_memlet(cb, kerR, t, "_in2", lin(ker_idx, {C, fftH, halfW}), cplx_ptr, dbg);
        builder.add_computational_memlet(cb, t, "_out", imgW, lin(iv, {N, C, fftH, halfW}), cplx_ptr, dbg);
    }

    // --- Inverse 2D FFT back to a real buffer. ---
    {
        auto& blk = builder.add_block(new_sequence, {}, dbg);
        auto& fnode = builder.add_library_node<math::tensor::C2RFFT2DNode>(
            blk,
            dbg,
            cuda::ImplementationType_CUDAWithTransfers,
            std::vector<symbolic::Expression>{NC, fftH, fftW},
            types::PrimitiveType::Float
        );
        auto& yacc = builder.add_access(blk, invReal, dbg);
        builder.add_computational_memlet(blk, yacc, fnode, "Y", {}, real_ptr, dbg);
        auto& xacc = builder.add_access(blk, specImg, dbg);
        builder.add_computational_memlet(blk, xacc, fnode, "X", {}, cplx_ptr, dbg);
    }

    // --- Crop (+ optional bias) into the output window. ---
    {
        auto [seq, iv] = add_maps(new_sequence, {N, C, Hh, Ww});
        auto& cb = builder.add_block(*seq, {}, dbg);
        auto& inacc = builder.add_access(cb, invReal, dbg);
        auto& yacc = builder.add_access(cb, b.access_Y->data(), dbg);
        // InvReal[n, c, r + cropH, w + cropW].
        std::vector<symbolic::Expression> in_idx = {
            iv[0], iv[1], symbolic::add(iv[2], cropH), symbolic::add(iv[3], cropW)
        };
        if (b.has_bias) {
            auto& bacc = builder.add_access(cb, b.access_B->data(), dbg);
            auto& t = builder.add_tasklet(cb, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, dbg);
            builder.add_computational_memlet(cb, inacc, t, "_in1", lin(in_idx, {N, C, fftH, fftW}), real_ptr, dbg);
            builder.add_computational_memlet(cb, bacc, t, "_in2", lin({iv[1]}, {C}), real_ptr, dbg);
            builder.add_computational_memlet(cb, t, "_out", yacc, lin(iv, {N, C, Hh, Ww}), real_ptr, dbg);
        } else {
            auto& t = builder.add_tasklet(cb, data_flow::TaskletCode::assign, "_out", {"_in"}, dbg);
            builder.add_computational_memlet(cb, inacc, t, "_in", lin(in_idx, {N, C, fftH, fftW}), real_ptr, dbg);
            builder.add_computational_memlet(cb, t, "_out", yacc, lin(iv, {N, C, Hh, Ww}), real_ptr, dbg);
        }
    }

    // --- Free the transients. ---
    for (const auto& buf : {padImg, padKer, invReal}) {
        stdlib::add_free_block(builder, new_sequence, buf, real_ptr, dbg);
    }
    for (const auto& buf : {specImg, specKer}) {
        stdlib::add_free_block(builder, new_sequence, buf, cplx_ptr, dbg);
    }

    // Remove the original convolution block.
    builder.clear_code_node_legacy(*b.block, node);
    builder.remove_child(*b.block_parent, b.block_index + 1);

    return true;
}

} // namespace expanders
} // namespace sdfg
