#include "sdfg/expanders/conv_fft_expander.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace expanders {

using structured_control_flow::Sequence;

bool ConvFFTExpander::is_applicable(const math::tensor::ConvNode& node) {
    // Only 2D convolutions.
    if (node.kernel_shape().size() != 2) {
        return false;
    }
    // Depthwise: one filter per input channel, one output channel per input channel.
    const auto& channels = node.shape()[1];
    if (!symbolic::eq(node.group(), channels)) {
        return false;
    }
    if (!symbolic::eq(node.output_channels(), channels)) {
        return false;
    }
    // Unit strides and dilations (FFT convolution assumes dense stride-1 sampling).
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
    return true;
}

bool ConvFFTExpander::enabled() {
    auto env = getenv("DOCC_CONV_FFT");
    if (env == nullptr) {
        return false;
    }
    std::string env_str(env);
    std::transform(env_str.begin(), env_str.end(), env_str.begin(), ::tolower);
    return env_str == "1" || env_str == "true";
}

bool ConvFFTExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return expand_conv_fft(builder, analysis_manager, node_);
}

bool ConvFFTExpander::expand_conv_fft(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, math::tensor::ConvNode& node
) {
    if (!enabled()) {
        return false;
    }
    if (!is_applicable(node)) {
        return false;
    }

    auto& dfg = node.get_parent();
    math::tensor::ConvNode::ConvExpandPrerequisits b;
    if (!node.check_expandable(dfg, b)) {
        return false;
    }

    auto prim = node.primitive_type(dfg);
    if (prim != types::PrimitiveType::Float && prim != types::PrimitiveType::Double) {
        return false;
    }
    auto cplx_prim = prim == types::PrimitiveType::Double ? types::PrimitiveType::CDouble
                                                          : types::PrimitiveType::CFloat;

    const DebugInfo& dbg = node.debug_info();

    types::Scalar base_type(prim);
    types::Scalar cplx_type(cplx_prim);
    types::Pointer real_ptr(base_type);
    types::Pointer cplx_ptr(cplx_type);
    types::Scalar indvar_type(types::PrimitiveType::Int64);

    // Geometry.
    const auto N = node.shape()[0];
    const auto C = node.shape()[1];
    const auto H = node.shape()[2];
    const auto W = node.shape()[3];
    const auto Kh = node.kernel_shape()[0];
    const auto Kw = node.kernel_shape()[1];
    const auto pad_h = node.pads()[0];
    const auto pad_w = node.pads()[1];

    // Linear-convolution padded extents and Hermitian last dim.
    const auto padH = symbolic::sub(symbolic::add(H, Kh), symbolic::one());
    const auto padW = symbolic::sub(symbolic::add(W, Kw), symbolic::one());
    const auto cW = symbolic::add(symbolic::div(padW, symbolic::integer(2)), symbolic::one());
    // Crop offset that recovers the correlation window: (K - 1 - pad_begin).
    const auto crop_h = symbolic::sub(symbolic::sub(Kh, symbolic::one()), pad_h);
    const auto crop_w = symbolic::sub(symbolic::sub(Kw, symbolic::one()), pad_w);

    const auto NC = symbolic::mul(N, C);
    const auto out_shape = node.get_out_shape();

    // New sequence that replaces the convolution block.
    auto& new_sequence = builder.add_sequence_before(
        *b.block_parent, *b.block, b.block_parent->at(b.block_index).second.assignments(), b.block->debug_info()
    );

    // Allocate the intermediate host buffers.
    auto make_buffer = [&](const std::string& prefix, const types::Scalar& elem, const symbolic::Expression& count) {
        types::Pointer ptr(elem);
        auto name = builder.find_new_name(prefix);
        builder.add_container(name, ptr);
        stdlib::add_malloc_block(builder, new_sequence, name, symbolic::mul(count, symbolic::size_of_type(elem)), ptr, dbg);
        return name;
    };

    const auto real_count = symbolic::mul(NC, symbolic::mul(padH, padW));
    const auto cplx_count = symbolic::mul(NC, symbolic::mul(padH, cW));
    const auto w_real_count = symbolic::mul(C, symbolic::mul(padH, padW));
    const auto w_cplx_count = symbolic::mul(C, symbolic::mul(padH, cW));

    auto xpad = make_buffer("_fft_xpad", base_type, real_count);
    auto wpad = make_buffer("_fft_wpad", base_type, w_real_count);
    auto wsrc = make_buffer("_fft_wsrc", base_type, symbolic::mul(C, symbolic::mul(Kh, Kw)));
    auto fx = make_buffer("_fft_fx", cplx_type, cplx_count);
    auto fw = make_buffer("_fft_fw", cplx_type, w_cplx_count);
    auto fy = make_buffer("_fft_fy", cplx_type, cplx_count);
    auto ifft_out = make_buffer("_fft_ifft_out", base_type, real_count);

    // Loop helper.
    auto add_loop = [&](Sequence*& seq, const char* prefix, const symbolic::Expression& bound) {
        auto name = builder.find_new_name(prefix);
        builder.add_container(name, indvar_type);
        auto sym = symbolic::symbol(name);
        auto& loop = builder.add_map(
            *seq,
            sym,
            symbolic::Lt(sym, bound),
            symbolic::zero(),
            symbolic::add(sym, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            dbg
        );
        seq = &loop.root();
        return sym;
    };

    // ---- 1. Zero-pad the input into xpad (offset 0). ------------------------------------
    {
        Sequence* seq = &new_sequence;
        auto n = add_loop(seq, "_n", N);
        auto c = add_loop(seq, "_c", C);
        auto i = add_loop(seq, "_i", padH);
        auto j = add_loop(seq, "_j", padW);

        auto nc = symbolic::add(symbolic::mul(n, C), c);
        auto flat_x = symbolic::add(symbolic::mul(symbolic::add(symbolic::mul(nc, padH), i), padW), j);

        auto cond = symbolic::And(symbolic::Lt(i, H), symbolic::Lt(j, W));
        auto& branch = builder.add_if_else(*seq, {}, dbg);
        auto& in_seq = builder.add_case(branch, cond, dbg);
        auto& out_seq = builder.add_case(branch, symbolic::Not(cond), dbg);
        {
            auto& blk = builder.add_block(in_seq, {}, dbg);
            auto& xacc = builder.add_access(blk, b.access_X->data(), b.access_X->debug_info());
            auto& dst = builder.add_access(blk, xpad, dbg);
            auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::assign, "_out", {"_in"}, dbg);
            builder.add_computational_memlet(blk, xacc, t, "_in", {n, c, i, j}, b.iedge_X->base_type(), dbg);
            builder.add_computational_memlet(blk, t, "_out", dst, {flat_x}, real_ptr, dbg);
        }
        {
            auto& blk = builder.add_block(out_seq, {}, dbg);
            auto& zero = builder.add_constant(blk, "0.0", base_type, dbg);
            auto& dst = builder.add_access(blk, xpad, dbg);
            auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::assign, "_out", {"_in"}, dbg);
            builder.add_computational_memlet(blk, zero, t, "_in", {}, base_type, dbg);
            builder.add_computational_memlet(blk, t, "_out", dst, {flat_x}, real_ptr, dbg);
        }
    }

    // ---- 1b. Copy the weight into a contiguous intermediate buffer. The pad below then
    //          reads a known-size malloc buffer instead of the raw weight argument, which
    //          keeps it offloadable after loop collapsing rewrites accesses with div/mod
    //          (argument-size analysis can only bound intermediate malloc buffers, not the
    //          collapsed access range of a function argument). -----------------------------
    {
        Sequence* seq = &new_sequence;
        auto c = add_loop(seq, "_c", C);
        auto ki = add_loop(seq, "_ki", Kh);
        auto kj = add_loop(seq, "_kj", Kw);
        auto flat_ws = symbolic::add(symbolic::mul(symbolic::add(symbolic::mul(c, Kh), ki), Kw), kj);
        auto& blk = builder.add_block(*seq, {}, dbg);
        auto& wacc = builder.add_access(blk, b.access_W->data(), b.access_W->debug_info());
        auto& dst = builder.add_access(blk, wsrc, dbg);
        auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::assign, "_out", {"_in"}, dbg);
        builder
            .add_computational_memlet(blk, wacc, t, "_in", {c, symbolic::zero(), ki, kj}, b.iedge_W->base_type(), dbg);
        builder.add_computational_memlet(blk, t, "_out", dst, {flat_ws}, real_ptr, dbg);
    }

    // ---- 2. Zero-pad and flip the weight into wpad (offset 0). --------------------------
    {
        Sequence* seq = &new_sequence;
        auto c = add_loop(seq, "_c", C);
        auto i = add_loop(seq, "_i", padH);
        auto j = add_loop(seq, "_j", padW);

        auto flat_w = symbolic::add(symbolic::mul(symbolic::add(symbolic::mul(c, padH), i), padW), j);
        auto wi = symbolic::sub(symbolic::sub(Kh, symbolic::one()), i);
        auto wj = symbolic::sub(symbolic::sub(Kw, symbolic::one()), j);

        auto cond = symbolic::And(symbolic::Lt(i, Kh), symbolic::Lt(j, Kw));
        auto& branch = builder.add_if_else(*seq, {}, dbg);
        auto& in_seq = builder.add_case(branch, cond, dbg);
        auto& out_seq = builder.add_case(branch, symbolic::Not(cond), dbg);
        {
            auto& blk = builder.add_block(in_seq, {}, dbg);
            auto& wacc = builder.add_access(blk, wsrc, dbg);
            auto& dst = builder.add_access(blk, wpad, dbg);
            auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::assign, "_out", {"_in"}, dbg);
            auto flat_ws = symbolic::add(symbolic::mul(symbolic::add(symbolic::mul(c, Kh), wi), Kw), wj);
            builder.add_computational_memlet(blk, wacc, t, "_in", {flat_ws}, real_ptr, dbg);
            builder.add_computational_memlet(blk, t, "_out", dst, {flat_w}, real_ptr, dbg);
        }
        {
            auto& blk = builder.add_block(out_seq, {}, dbg);
            auto& zero = builder.add_constant(blk, "0.0", base_type, dbg);
            auto& dst = builder.add_access(blk, wpad, dbg);
            auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::assign, "_out", {"_in"}, dbg);
            builder.add_computational_memlet(blk, zero, t, "_in", {}, base_type, dbg);
            builder.add_computational_memlet(blk, t, "_out", dst, {flat_w}, real_ptr, dbg);
        }
    }

    std::vector<symbolic::Expression> transform_shape{padH, padW};

    // ---- 3. Forward FFT of the padded input: xpad -> fx. -------------------------------
    {
        auto& blk = builder.add_block(new_sequence, {}, dbg);
        auto& x_acc = builder.add_access(blk, xpad, dbg);
        auto& fx_acc = builder.add_access(blk, fx, dbg);
        auto& fft = builder.add_library_node<
            math::tensor::FFTNode>(blk, dbg, data_flow::ImplementationType_NONE, transform_shape, NC, prim);
        builder.add_computational_memlet(blk, x_acc, fft, "__X", {}, real_ptr, dbg);
        builder.add_computational_memlet(blk, fx_acc, fft, "__Y", {}, cplx_ptr, dbg);
    }

    // ---- 4. Forward FFT of the padded weight: wpad -> fw. ------------------------------
    {
        auto& blk = builder.add_block(new_sequence, {}, dbg);
        auto& w_acc = builder.add_access(blk, wpad, dbg);
        auto& fw_acc = builder.add_access(blk, fw, dbg);
        auto& fft = builder.add_library_node<
            math::tensor::FFTNode>(blk, dbg, data_flow::ImplementationType_NONE, transform_shape, C, prim);
        builder.add_computational_memlet(blk, w_acc, fft, "__X", {}, real_ptr, dbg);
        builder.add_computational_memlet(blk, fw_acc, fft, "__Y", {}, cplx_ptr, dbg);
    }

    // ---- 5. Frequency-domain pointwise complex multiply: fy = fx * fw (per channel). ---
    {
        Sequence* seq = &new_sequence;
        auto n = add_loop(seq, "_n", N);
        auto c = add_loop(seq, "_c", C);
        auto i = add_loop(seq, "_i", padH);
        auto j = add_loop(seq, "_j", cW);

        auto nc = symbolic::add(symbolic::mul(n, C), c);
        auto flat_fx = symbolic::add(symbolic::mul(symbolic::add(symbolic::mul(nc, padH), i), cW), j);
        auto flat_fw = symbolic::add(symbolic::mul(symbolic::add(symbolic::mul(c, padH), i), cW), j);

        auto& blk = builder.add_block(*seq, {}, dbg);
        auto& fx_acc = builder.add_access(blk, fx, dbg);
        auto& fw_acc = builder.add_access(blk, fw, dbg);
        auto& fy_acc = builder.add_access(blk, fy, dbg);
        auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::complex_mul, "_out", {"_in1", "_in2"}, dbg);
        builder.add_computational_memlet(blk, fx_acc, t, "_in1", {flat_fx}, cplx_ptr, dbg);
        builder.add_computational_memlet(blk, fw_acc, t, "_in2", {flat_fw}, cplx_ptr, dbg);
        builder.add_computational_memlet(blk, t, "_out", fy_acc, {flat_fx}, cplx_ptr, dbg);
    }

    // ---- 6. Inverse FFT: fy -> ifft_out. ----------------------------------------------
    {
        auto& blk = builder.add_block(new_sequence, {}, dbg);
        auto& fy_acc = builder.add_access(blk, fy, dbg);
        auto& out_acc = builder.add_access(blk, ifft_out, dbg);
        auto& ifft = builder.add_library_node<
            math::tensor::IFFTNode>(blk, dbg, data_flow::ImplementationType_NONE, transform_shape, NC, prim);
        builder.add_computational_memlet(blk, fy_acc, ifft, "__X", {}, cplx_ptr, dbg);
        builder.add_computational_memlet(blk, out_acc, ifft, "__Y", {}, real_ptr, dbg);
    }

    // ---- 7. Crop, normalize by (padH*padW), and add bias. -----------------------------
    {
        Sequence* seq = &new_sequence;
        auto n = add_loop(seq, "_n", N);
        auto c = add_loop(seq, "_c", C);
        auto r = add_loop(seq, "_r", out_shape[0]);
        auto w = add_loop(seq, "_w", out_shape[1]);

        auto nc = symbolic::add(symbolic::mul(n, C), c);
        auto ii = symbolic::add(r, crop_h);
        auto jj = symbolic::add(w, crop_w);
        auto flat_ifft = symbolic::add(symbolic::mul(symbolic::add(symbolic::mul(nc, padH), ii), padW), jj);
        auto scale_expr = symbolic::mul(padH, padW);

        auto& blk = builder.add_block(*seq, {}, dbg);
        auto& ifft_acc = builder.add_access(blk, ifft_out, dbg);
        auto& scale = builder.add_constant(blk, scale_expr->__str__(), base_type, dbg);
        auto& div = builder.add_tasklet(blk, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"}, dbg);
        builder.add_computational_memlet(blk, ifft_acc, div, "_in1", {flat_ifft}, real_ptr, dbg);
        builder.add_computational_memlet(blk, scale, div, "_in2", {}, base_type, dbg);

        data_flow::Subset y_subset{n, c, r, w};
        if (b.has_bias) {
            auto scaled = builder.find_new_name("_fft_scaled");
            builder.add_container(scaled, base_type);
            auto& scaled_w = builder.add_access(blk, scaled, dbg);
            builder.add_computational_memlet(blk, div, "_out", scaled_w, {}, base_type, dbg);

            auto& scaled_r = builder.add_access(blk, scaled, dbg);
            auto& bias_acc = builder.add_access(blk, b.access_B->data(), b.access_B->debug_info());
            auto& y_acc = builder.add_access(blk, b.access_Y->data(), b.access_Y->debug_info());
            auto& add = builder.add_tasklet(blk, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, dbg);
            builder.add_computational_memlet(blk, scaled_r, add, "_in1", {}, base_type, dbg);
            builder.add_computational_memlet(blk, bias_acc, add, "_in2", {c}, b.iedge_B->base_type(), dbg);
            builder.add_computational_memlet(blk, add, "_out", y_acc, y_subset, b.iedge_Y->base_type(), dbg);
        } else {
            auto& y_acc = builder.add_access(blk, b.access_Y->data(), b.access_Y->debug_info());
            builder.add_computational_memlet(blk, div, "_out", y_acc, y_subset, b.iedge_Y->base_type(), dbg);
        }
    }

    // ---- Free the intermediate buffers. -----------------------------------------------
    for (const auto& buf : {xpad, wpad, wsrc, fx, fw, fy, ifft_out}) {
        const auto& buf_type = builder.subject().type(buf);
        stdlib::add_free_block(builder, new_sequence, buf, buf_type, dbg);
    }

    // Remove the original convolution block.
    builder.clear_code_node_legacy(*b.block, node);
    builder.remove_child(*b.block_parent, b.block_index + 1);

    return true;
}

} // namespace expanders
} // namespace sdfg
