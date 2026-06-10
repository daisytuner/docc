// Regression tests for CUDATransform / OffloadTransform on the im2col pattern
// produced by ResNet's first stride-2 7x7 conv lowering.
//
// The map writes a `_patches` buffer from an input image `_1` and previously
// (commit prior to the regression observed in resnet `__docc_GraphModule.cpp`)
// was offloaded to a single CUDA kernel. It is now correctly offloaded again
// thanks to the coupled-constraint upper bound, offset-aware delinearize
// stride check, and the sub-dominant stride merge in `delinearize`.
//
// Two tests:
//   * `CollapsedTwoDimMap` - exact shape produced by the optimizer: two maps
//     over collapsed indvars with mod/div arithmetic in the memlet subsets.
//   * `ExplicitSixDimMap` - the logically equivalent un-collapsed form (six
//     nested maps with simple affine subscripts). Useful to disentangle
//     whether the regression is in subset analysis under collapsed indvars or
//     in the offload-transform criteria themselves.

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/transformations/offloading/cuda_transform.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg::cuda {

namespace {

// Constants mirroring the failing resnet kernel.
constexpr int kN = 32;
constexpr int kCin = 3;
constexpr int kHin = 224;
constexpr int kHout = 112;
constexpr int kKh = 7;

constexpr int kCollapsedOuter = kN * kHout * kHout; // 401408
constexpr int kCollapsedInner = kCin * kKh * kKh; // 147

constexpr int kStrideNCin = kHin * kHin; // 50176
constexpr int kStrideN = kCin * kHin * kHin; // 150528

constexpr int kStridePatchN = kHout * kHout * kCin * kKh * kKh; // 1843968
constexpr int kStridePatchHout = kHout * kCin * kKh * kKh; // 16464
constexpr int kStridePatchWout = kCin * kKh * kKh; // 147
constexpr int kStridePatchC = kKh * kKh; // 49

// _patches0 size in elements: N * Hout * Wout * Cin * Kh * Kw
constexpr long long kPatchesElems = static_cast<long long>(kN) * kHout * kHout * kCin * kKh * kKh;
// _1 size in elements: N * Cin * Hin * Win
constexpr long long kImageElems = static_cast<long long>(kN) * kCin * kHin * kHin;

symbolic::Expression i(long long v) { return symbolic::integer(v); }
symbolic::Symbol s(const std::string& n) { return symbolic::symbol(n); }

} // namespace

TEST(CudaTransformIm2colTest, CollapsedTwoDimMap) {
    builder::StructuredSDFGBuilder builder("im2col_collapsed", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer f32ptr(f32);
    types::Scalar i64(types::PrimitiveType::Int64);

    builder.add_container("_n0_collapsed0", i64);
    builder.add_container("_c0_collapsed0", i64);
    builder.add_container("_1", f32ptr, /*is_argument=*/true);
    builder.add_container("_patches0", f32ptr, /*is_argument=*/true);

    ScheduleType seq = ScheduleType_Sequential::create();

    auto& outer_map = builder.add_map(
        root,
        s("_n0_collapsed0"),
        symbolic::Lt(s("_n0_collapsed0"), i(kCollapsedOuter)),
        i(0),
        symbolic::add(s("_n0_collapsed0"), i(1)),
        seq
    );
    auto& inner_map = builder.add_map(
        outer_map.root(),
        s("_c0_collapsed0"),
        symbolic::Lt(s("_c0_collapsed0"), i(kCollapsedInner)),
        i(0),
        symbolic::add(s("_c0_collapsed0"), i(1)),
        seq
    );

    // Helpers
    auto kh_mod = symbolic::mod(symbolic::div(s("_c0_collapsed0"), i(kKh)), i(kKh));
    auto kw_mod = symbolic::mod(s("_c0_collapsed0"), i(kKh));
    auto hout_mod = symbolic::mod(symbolic::div(s("_n0_collapsed0"), i(kHout)), i(kHout));
    auto wout_mod = symbolic::mod(s("_n0_collapsed0"), i(kHout));
    auto c_div = symbolic::div(s("_c0_collapsed0"), i(kStridePatchC)); // c0 / 49
    auto n_div = symbolic::div(s("_n0_collapsed0"), i(kHout * kHout)); // n0 / 12544

    // h_in = -3 + ((c0/7)%7) + 2*((n0/112)%112)
    auto h_in = symbolic::add(i(-(kKh / 2)), symbolic::add(kh_mod, symbolic::mul(i(2), hout_mod)));
    // w_in = -3 + (c0%7) + 2*(n0%112)
    auto w_in = symbolic::add(i(-(kKh / 2)), symbolic::add(kw_mod, symbolic::mul(i(2), wout_mod)));

    auto cond_in_bounds = symbolic::
        And(symbolic::And(symbolic::Ge(w_in, i(0)), symbolic::Ge(h_in, i(0))),
            symbolic::And(symbolic::Lt(w_in, i(kHin)), symbolic::Lt(h_in, i(kHin))));
    auto cond_out_of_bounds = symbolic::
        Or(symbolic::Or(symbolic::Ge(w_in, i(kHin)), symbolic::Ge(h_in, i(kHin))),
           symbolic::Or(symbolic::Lt(w_in, i(0)), symbolic::Lt(h_in, i(0))));

    auto& ifelse = builder.add_if_else(inner_map.root());
    auto& case_in = builder.add_case(ifelse, cond_in_bounds);
    auto& case_out = builder.add_case(ifelse, cond_out_of_bounds);

    // out_idx = 49*(c0/49) + 1843968*(n0/12544) + (c0%7) + 147*(n0%112)
    //         + 7*((c0/7)%7) + 16464*((n0/112)%112)
    auto out_idx = symbolic::
        add(symbolic::
                add(symbolic::add(symbolic::mul(i(kStridePatchC), c_div), symbolic::mul(i(kStridePatchN), n_div)),
                    symbolic::add(kw_mod, symbolic::mul(i(kStridePatchWout), wout_mod))),
            symbolic::add(symbolic::mul(i(kKh), kh_mod), symbolic::mul(i(kStridePatchHout), hout_mod)));

    // in_idx = -3 + 224*(-3 + ((c0/7)%7) + 2*((n0/112)%112))
    //        + 50176*(c0/49) + 150528*(n0/12544) + (c0%7) + 2*(n0%112)
    auto in_idx = symbolic::add(
        i(-(kKh / 2)),
        symbolic::
            add(symbolic::add(symbolic::mul(i(kHin), h_in), symbolic::mul(i(kStrideNCin), c_div)),
                symbolic::add(symbolic::mul(i(kStrideN), n_div), symbolic::add(kw_mod, symbolic::mul(i(2), wout_mod))))
    );

    // In-bounds branch: _patches0[out_idx] = _1[in_idx]
    {
        auto& block = builder.add_block(case_in);
        auto& read = builder.add_access(block, "_1");
        auto& write = builder.add_access(block, "_patches0");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
        builder.add_computational_memlet(block, read, tasklet, "in_", {in_idx});
        builder.add_computational_memlet(block, tasklet, "out_", write, {out_idx});
    }
    // Out-of-bounds branch: _patches0[out_idx] = 0
    {
        auto& block = builder.add_block(case_out);
        auto& write = builder.add_access(block, "_patches0");
        auto& constant = builder.add_constant(block, "0.0f", f32);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
        builder.add_computational_memlet(block, constant, tasklet, "in_", {}, f32);
        builder.add_computational_memlet(block, tasklet, "out_", write, {out_idx});
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDATransform transform(outer_map, /*block_size=*/32);

    // The outer map of the collapsed im2col pattern must be recognised as
    // offloadable to a single CUDA kernel.
    EXPECT_TRUE(transform.can_be_applied(builder, analysis_manager))
        << "OffloadTransform should accept the collapsed im2col map.";
}

TEST(CudaTransformIm2colTest, ExplicitSixDimMap) {
    builder::StructuredSDFGBuilder builder("im2col_explicit", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer f32ptr(f32);
    types::Scalar i64(types::PrimitiveType::Int64);

    builder.add_container("n", i64);
    builder.add_container("hout", i64);
    builder.add_container("wout", i64);
    builder.add_container("c", i64);
    builder.add_container("kh", i64);
    builder.add_container("kw", i64);
    builder.add_container("_1", f32ptr, /*is_argument=*/true);
    builder.add_container("_patches0", f32ptr, /*is_argument=*/true);

    ScheduleType seq = ScheduleType_Sequential::create();

    auto add_simple_map = [&](structured_control_flow::Sequence& parent, const std::string& name, long long bound
                          ) -> structured_control_flow::Map& {
        return builder
            .add_map(parent, s(name), symbolic::Lt(s(name), i(bound)), i(0), symbolic::add(s(name), i(1)), seq);
    };

    auto& m_n = add_simple_map(root, "n", kN);
    auto& m_hout = add_simple_map(m_n.root(), "hout", kHout);
    auto& m_wout = add_simple_map(m_hout.root(), "wout", kHout);
    auto& m_c = add_simple_map(m_wout.root(), "c", kCin);
    auto& m_kh = add_simple_map(m_c.root(), "kh", kKh);
    auto& m_kw = add_simple_map(m_kh.root(), "kw", kKh);

    // h_in = 2*hout + kh - 3, w_in = 2*wout + kw - 3
    auto h_in = symbolic::sub(symbolic::add(symbolic::mul(i(2), s("hout")), s("kh")), i(kKh / 2));
    auto w_in = symbolic::sub(symbolic::add(symbolic::mul(i(2), s("wout")), s("kw")), i(kKh / 2));

    auto cond_in_bounds = symbolic::
        And(symbolic::And(symbolic::Ge(w_in, i(0)), symbolic::Ge(h_in, i(0))),
            symbolic::And(symbolic::Lt(w_in, i(kHin)), symbolic::Lt(h_in, i(kHin))));
    auto cond_out_of_bounds = symbolic::
        Or(symbolic::Or(symbolic::Ge(w_in, i(kHin)), symbolic::Ge(h_in, i(kHin))),
           symbolic::Or(symbolic::Lt(w_in, i(0)), symbolic::Lt(h_in, i(0))));

    auto& ifelse = builder.add_if_else(m_kw.root());
    auto& case_in = builder.add_case(ifelse, cond_in_bounds);
    auto& case_out = builder.add_case(ifelse, cond_out_of_bounds);

    auto out_idx = symbolic::add(
        symbolic::
            add(symbolic::add(symbolic::mul(i(kStridePatchN), s("n")), symbolic::mul(i(kStridePatchHout), s("hout"))),
                symbolic::add(symbolic::mul(i(kStridePatchWout), s("wout")), symbolic::mul(i(kStridePatchC), s("c")))),
        symbolic::add(symbolic::mul(i(kKh), s("kh")), s("kw"))
    );
    auto in_idx = symbolic::
        add(symbolic::
                add(symbolic::add(symbolic::mul(i(kStrideN), s("n")), symbolic::mul(i(kStrideNCin), s("c"))),
                    symbolic::mul(i(kHin), h_in)),
            w_in);

    {
        auto& block = builder.add_block(case_in);
        auto& read = builder.add_access(block, "_1");
        auto& write = builder.add_access(block, "_patches0");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
        builder.add_computational_memlet(block, read, tasklet, "in_", {in_idx});
        builder.add_computational_memlet(block, tasklet, "out_", write, {out_idx});
    }
    {
        auto& block = builder.add_block(case_out);
        auto& write = builder.add_access(block, "_patches0");
        auto& constant = builder.add_constant(block, "0.0f", f32);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
        builder.add_computational_memlet(block, constant, tasklet, "in_", {}, f32);
        builder.add_computational_memlet(block, tasklet, "out_", write, {out_idx});
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDATransform transform(m_n, /*block_size=*/32);

    EXPECT_TRUE(transform.can_be_applied(builder, analysis_manager))
        << "OffloadTransform should accept the explicit (un-collapsed) im2col map.";
}

} // namespace sdfg::cuda
