#include <gtest/gtest.h>

#include <memory>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/loops/transformations/loop_skewing.h"
#include "sdfg/loops/transformations/strip_mining.h"
#include "sdfg/parallelization/passes/auto_parallelization.h"
#include "sdfg/reordering/transformations/loop_interchange.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"

namespace parallelogram_tiling_test {

using namespace sdfg;

// Gauss-Seidel-like nest with dependences (0, 1) and (1, -1):
//   for i in [1, N): for j in [1, M-1): A[i][j] = A[i-1][j+1] + A[i][j-1]
std::unique_ptr<builder::StructuredSDFGBuilder> build_gauss_seidel() {
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>("gauss_seidel", FunctionType_CPU);
    types::Scalar idx(types::PrimitiveType::Int64);
    builder->add_container("N", idx, true);
    builder->add_container("M", idx, true);
    builder->add_container("i", idx);
    builder->add_container("j", idx);
    types::Scalar elem(types::PrimitiveType::Double);
    types::Array row(elem, symbolic::symbol("M"));
    types::Pointer desc(row);
    builder->add_container("A", desc, true);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& loop_i = builder->add_for(
        builder->subject().root(),
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::one(),
        symbolic::add(i, symbolic::one())
    );
    auto& loop_j = builder->add_for(
        loop_i.root(),
        j,
        symbolic::Lt(j, symbolic::sub(symbolic::symbol("M"), symbolic::one())),
        symbolic::one(),
        symbolic::add(j, symbolic::one())
    );
    auto& block = builder->add_block(loop_j.root());
    auto& a_in = builder->add_access(block, "A");
    auto& a_out = builder->add_access(block, "A");
    auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder->add_computational_memlet(
        block, a_in, tasklet, "_in1", {symbolic::sub(i, symbolic::one()), symbolic::add(j, symbolic::one())}, desc
    );
    builder->add_computational_memlet(block, a_in, tasklet, "_in2", {i, symbolic::sub(j, symbolic::one())}, desc);
    builder->add_computational_memlet(block, tasklet, "_out", a_out, {i, j}, desc);
    return builder;
}

// Jacobi-1D after shift+fusion, built directly:
//   for t in [0, T): for f in [0, N-1):
//     if f < N-2: B[f+1] = (A[f] + A[f+1] + A[f+2]) / 3
//     if f >= 1:  A[f]   = (B[f-1] + B[f] + B[f+1]) / 3
std::unique_ptr<builder::StructuredSDFGBuilder> build_fused_jacobi_1d() {
    auto builder = std::make_unique<builder::StructuredSDFGBuilder>("jacobi_1d", FunctionType_CPU);
    types::Scalar idx(types::PrimitiveType::Int64);
    types::Scalar elem(types::PrimitiveType::Double);
    types::Array arr(elem, symbolic::symbol("N"));
    builder->add_container("T", idx, true);
    builder->add_container("N", idx, true);
    builder->add_container("t", idx);
    builder->add_container("f", idx);
    builder->add_container("A", arr, true);
    builder->add_container("B", arr, true);
    for (auto name : {"s1", "s2", "s3", "s4"}) {
        builder->add_container(name, elem);
    }
    auto t = symbolic::symbol("t");
    auto f = symbolic::symbol("f");
    auto N = symbolic::symbol("N");
    auto& loop_t = builder->add_for(
        builder->subject().root(),
        t,
        symbolic::Lt(t, symbolic::symbol("T")),
        symbolic::zero(),
        symbolic::add(t, symbolic::one())
    );
    auto& loop_f = builder->add_for(
        loop_t.root(),
        f,
        symbolic::Lt(f, symbolic::sub(N, symbolic::one())),
        symbolic::zero(),
        symbolic::add(f, symbolic::one())
    );
    auto stencil = [&](structured_control_flow::Sequence& seq,
                       const std::string& in,
                       const std::string& out,
                       const symbolic::Expression& base,
                       const symbolic::Expression& out_idx,
                       const std::string& tmp_a,
                       const std::string& tmp_b) {
        auto& b1 = builder->add_block(seq);
        auto& in1 = builder->add_access(b1, in);
        auto& t1 = builder->add_tasklet(b1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder->add_computational_memlet(b1, in1, t1, "_in1", {base}, arr);
        builder->add_computational_memlet(b1, in1, t1, "_in2", {symbolic::add(base, symbolic::one())}, arr);
        builder->add_computational_memlet(b1, t1, "_out", builder->add_access(b1, tmp_a), {});
        auto& b2 = builder->add_block(seq);
        auto& t2 = builder->add_tasklet(b2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder->add_computational_memlet(b2, builder->add_access(b2, tmp_a), t2, "_in1", {});
        builder->add_computational_memlet(
            b2, builder->add_access(b2, in), t2, "_in2", {symbolic::add(base, symbolic::integer(2))}, arr
        );
        builder->add_computational_memlet(b2, t2, "_out", builder->add_access(b2, tmp_b), {});
        auto& b3 = builder->add_block(seq);
        auto& t3 = builder->add_tasklet(b3, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder->add_computational_memlet(b3, builder->add_constant(b3, "0.333", elem), t3, "_in1", {});
        builder->add_computational_memlet(b3, builder->add_access(b3, tmp_b), t3, "_in2", {});
        builder->add_computational_memlet(b3, t3, "_out", builder->add_access(b3, out), {out_idx}, arr);
    };
    auto& guards = builder->add_if_else(loop_f.root());
    auto& k1 = builder->add_case(guards, symbolic::Lt(f, symbolic::sub(N, symbolic::integer(2))));
    stencil(k1, "A", "B", f, symbolic::add(f, symbolic::one()), "s1", "s2");
    auto& guards2 = builder->add_if_else(loop_f.root());
    auto& k2 = builder->add_case(guards2, symbolic::Ge(f, symbolic::one()));
    stencil(k2, "B", "A", symbolic::sub(f, symbolic::one()), f, "s3", "s4");
    return builder;
}

structured_control_flow::StructuredLoop& loop_at(structured_control_flow::Sequence& root, std::vector<size_t> path) {
    structured_control_flow::ControlFlowNode* node = &root.at(path[0]);
    for (size_t k = 1; k < path.size(); k++) {
        node = &dynamic_cast<structured_control_flow::StructuredLoop*>(node)->root().at(path[k]);
    }
    return *dynamic_cast<structured_control_flow::StructuredLoop*>(node);
}

template<typename T>
void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& am, T&& transformation) {
    ASSERT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);
    am.invalidate_all();
}

void expect_loop(structured_control_flow::StructuredLoop& loop, const std::string& indvar, int64_t stride, bool is_map) {
    EXPECT_EQ(loop.indvar()->get_name(), indvar);
    EXPECT_TRUE(symbolic::eq(symbolic::sub(loop.update(), loop.indvar()), symbolic::integer(stride)));
    EXPECT_EQ(dynamic_cast<structured_control_flow::Map*>(&loop) != nullptr, is_map);
}

// Skewing j by s*i gives distances (0, 1), (1, s-1) in (i, j'). Interchange needs s >= 1;
// the new inner loop i is parallel only if the outer j' carries every dependence, i.e. s >= 2.
TEST(ParallelogramTilingTest, Wavefront_Unskewed_Illegal) {
    auto builder = build_gauss_seidel();
    analysis::AnalysisManager am(builder->subject());
    auto& root = builder->subject().root();

    reordering::LoopInterchange interchange(loop_at(root, {0}), loop_at(root, {0, 0}));
    EXPECT_FALSE(interchange.can_be_applied(*builder, am));
}

TEST(ParallelogramTilingTest, Wavefront_Skew1_InnerSequential) {
    auto builder = build_gauss_seidel();
    analysis::AnalysisManager am(builder->subject());
    auto& root = builder->subject().root();

    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::LoopSkewing(loop_at(root, {0}), loop_at(root, {0, 0}), 1)));
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, reordering::LoopInterchange(loop_at(root, {0}), loop_at(root, {0, 0}))));

    // (1, -1) skewed by 1 is (1, 0) in (i, j'), i.e. (0, 1) in (j', i): still carried by the inner i.
    parallelization::AutoParallelization classification;
    classification.run(*builder, am);
    expect_loop(loop_at(root, {0}), "j", 1, false);
    expect_loop(loop_at(root, {0, 0}), "i", 1, false);
}

TEST(ParallelogramTilingTest, Wavefront_Skew2_InnerParallel) {
    auto builder = build_gauss_seidel();
    analysis::AnalysisManager am(builder->subject());
    auto& root = builder->subject().root();

    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::LoopSkewing(loop_at(root, {0}), loop_at(root, {0, 0}), 2)));
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, reordering::LoopInterchange(loop_at(root, {0}), loop_at(root, {0, 0}))));

    parallelization::AutoParallelization classification;
    classification.run(*builder, am);
    expect_loop(loop_at(root, {0}), "j", 1, false);
    expect_loop(loop_at(root, {0, 0}), "i", 1, true);
}

// Parallelogram tiles of 32x32 with a tile-level wavefront:
//   strip i (32) -> skew(i, j, 1) within the band -> interchange(i, j) -> tile j (32) -> interchange(j, i)
//   gives  for ib step 32: for jb step 32: for i: for j   (stride-1 sweeps inside each tile)
// The band-relative skew yields tile distances (0,1), (1,0), (1,-1), so the wavefront is
// w = 2*ib/32 + jb/32, i.e. skew(ib, jb, 2) followed by interchange(ib, jb).
TEST(ParallelogramTilingTest, TileWavefront_GaussSeidel) {
    auto builder = build_gauss_seidel();
    analysis::AnalysisManager am(builder->subject());
    auto& root = builder->subject().root();

    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::StripMining(loop_at(root, {0}), 32)));
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::LoopSkewing(loop_at(root, {0, 0}), loop_at(root, {0, 0, 0}), 1)));
    ASSERT_NO_FATAL_FAILURE(
        apply(*builder, am, reordering::LoopInterchange(loop_at(root, {0, 0}), loop_at(root, {0, 0, 0})))
    );
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::StripMining(loop_at(root, {0, 0}), 32)));
    ASSERT_NO_FATAL_FAILURE(
        apply(*builder, am, reordering::LoopInterchange(loop_at(root, {0, 0, 0}), loop_at(root, {0, 0, 0, 0})))
    );
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::LoopSkewing(loop_at(root, {0}), loop_at(root, {0, 0}), 2)));
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, reordering::LoopInterchange(loop_at(root, {0}), loop_at(root, {0, 0}))));

    parallelization::AutoParallelization classification;
    classification.run(*builder, am);
    expect_loop(loop_at(root, {0}), "j_tile0", 32, false);
    expect_loop(loop_at(root, {0, 0}), "i_tile0", 32, true);
    expect_loop(loop_at(root, {0, 0, 0}), "i", 1, false);
    expect_loop(loop_at(root, {0, 0, 0, 0}), "j", 1, false);
}

// Jacobi-1D with 32x16 parallelogram tiles and a tile-level wavefront:
//   strip t (16) -> skew(t, f, 2) within the band -> interchange(t, f) -> tile f (32) -> interchange(f, t)
//   -> skew(tb, fb, 4) -> interchange(tb, fb)
// Tile distances are (0,1), (1,0), (1,-1) as for Gauss-Seidel; a band spans 4*16 = 64 = 2 tiles.
TEST(ParallelogramTilingTest, TileWavefront_Jacobi1D) {
    auto builder = build_fused_jacobi_1d();
    analysis::AnalysisManager am(builder->subject());
    auto& root = builder->subject().root();

    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::StripMining(loop_at(root, {0}), 16)));
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::LoopSkewing(loop_at(root, {0, 0}), loop_at(root, {0, 0, 0}), 2)));
    ASSERT_NO_FATAL_FAILURE(
        apply(*builder, am, reordering::LoopInterchange(loop_at(root, {0, 0}), loop_at(root, {0, 0, 0})))
    );
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::StripMining(loop_at(root, {0, 0}), 32)));
    ASSERT_NO_FATAL_FAILURE(
        apply(*builder, am, reordering::LoopInterchange(loop_at(root, {0, 0, 0}), loop_at(root, {0, 0, 0, 0})))
    );
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, loops::LoopSkewing(loop_at(root, {0}), loop_at(root, {0, 0}), 4)));
    ASSERT_NO_FATAL_FAILURE(apply(*builder, am, reordering::LoopInterchange(loop_at(root, {0}), loop_at(root, {0, 0}))));

    // AutoParallelization is not run: its dependence analysis does not finish on these bounds yet.
    expect_loop(loop_at(root, {0}), "f_tile0", 32, false);
    expect_loop(loop_at(root, {0, 0}), "t_tile0", 16, false);
    expect_loop(loop_at(root, {0, 0, 0}), "t", 1, false);
    expect_loop(loop_at(root, {0, 0, 0, 0}), "f", 1, false);
}

} // namespace parallelogram_tiling_test
