#include "sdfg/tiles/analysis/tile_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/targets/omp/schedule.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

// Sequential nest: no parallel axes, so the tile is private (registers).
TEST(TileAnalysisTest, Sequential_NoAxes) {
    builder::StructuredSDFGBuilder builder("ta_seq", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque;
    builder.add_container("i", sym);
    builder.add_container("t", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", opaque, true);

    auto i = symbolic::symbol("i");
    auto t = symbolic::symbol("t");
    auto& loop_i = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );
    auto& loop_t = builder.add_for(
        loop_i.root(),
        t,
        symbolic::Lt(t, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(t, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop_t.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tk, "_in", {t}, ptr);
    builder.add_computational_memlet(block, tk, "_out", c_out, {t}, ptr);

    analysis::AnalysisManager am(builder.subject());
    auto& ta = am.get<tiles::TileAnalysis>();
    const auto* tile = ta.tile(loop_t, "A");
    ASSERT_NE(tile, nullptr);
    EXPECT_TRUE(tile->axes().empty());
    EXPECT_FALSE(tile->cooperative());
    EXPECT_EQ(tile->required_space(), tiles::Space::Register);
    EXPECT_TRUE(tile->reads());
    // Source: contiguous extent-8 tile, offset 0.
    ASSERT_EQ(tile->source().rank(), 1u);
    EXPECT_TRUE(symbolic::eq(tile->source().shape()[0], symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(tile->source().stride()[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(tile->source().offset(), symbolic::integer(0)));
}

// GPU block map whose indvar addresses the tile base -> per-thread (registers).
TEST(TileAnalysisTest, GpuPerThread_PrivateRegisters) {
    builder::StructuredSDFGBuilder builder("ta_perthread", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque;
    builder.add_container("i", sym);
    builder.add_container("t", sym);
    builder.add_container("N", sym, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", opaque, true);

    auto i = symbolic::symbol("i");
    auto t = symbolic::symbol("t");
    auto N = symbolic::symbol("N");
    auto sched = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched, symbolic::integer(32));
    auto& map_i =
        builder
            .add_map(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched);
    auto& loop_t = builder.add_for(
        map_i.root(),
        t,
        symbolic::Lt(t, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(t, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop_t.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // A[i*8 + t]: base i*8 depends on i -> per-thread.
    auto idx = symbolic::add(symbolic::mul(i, symbolic::integer(8)), t);
    builder.add_computational_memlet(block, a_in, tk, "_in", {idx}, ptr);
    builder.add_computational_memlet(block, tk, "_out", c_out, {t}, ptr);

    analysis::AnalysisManager am(builder.subject());
    auto& ta = am.get<tiles::TileAnalysis>();
    const auto* tile = ta.tile(loop_t, "A");
    ASSERT_NE(tile, nullptr);
    ASSERT_EQ(tile->axes().size(), 1u);
    EXPECT_TRUE(tile->axes()[0].schedule().has_scratchpad());
    EXPECT_EQ(tile->axes()[0].role(), tiles::Role::Private);
    EXPECT_EQ(tile->axes()[0].schedule().level(), tiles::Level::Group);
    EXPECT_FALSE(tile->cooperative());
    EXPECT_EQ(tile->required_space(), tiles::Space::Register);
    // Source offset folds the per-thread base i*8; the varying dim is extent-8.
    // (MLA delinearizes into an extent-1 i-mode + extent-8 t-mode; coalesce drops it.)
    auto src = tiles::coalesce(tile->source());
    ASSERT_EQ(src.rank(), 1u);
    EXPECT_TRUE(symbolic::eq(src.shape()[0], symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(src.stride()[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(tile->source().offset(), symbolic::mul(i, symbolic::integer(8))));
}

// GPU block map whose indvar is absent from the tile base -> cooperative (shared).
TEST(TileAnalysisTest, GpuCooperative_SharedRead) {
    builder::StructuredSDFGBuilder builder("ta_coop", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque;
    builder.add_container("i", sym);
    builder.add_container("t", sym);
    builder.add_container("N", sym, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", opaque, true);

    auto i = symbolic::symbol("i");
    auto t = symbolic::symbol("t");
    auto N = symbolic::symbol("N");
    auto sched = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched, symbolic::integer(32));
    auto& map_i =
        builder
            .add_map(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched);
    auto& loop_t = builder.add_for(
        map_i.root(),
        t,
        symbolic::Lt(t, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(t, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop_t.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // A[t]: base independent of i -> all threads share the tile.
    builder.add_computational_memlet(block, a_in, tk, "_in", {t}, ptr);
    builder.add_computational_memlet(block, tk, "_out", c_out, {t}, ptr);

    analysis::AnalysisManager am(builder.subject());
    auto& ta = am.get<tiles::TileAnalysis>();
    const auto* tile = ta.tile(loop_t, "A");
    ASSERT_NE(tile, nullptr);
    ASSERT_EQ(tile->axes().size(), 1u);
    EXPECT_TRUE(tile->axes()[0].schedule().has_scratchpad());
    EXPECT_EQ(tile->axes()[0].role(), tiles::Role::Cooperative);
    EXPECT_EQ(tile->axes()[0].schedule().level(), tiles::Level::Group);
    EXPECT_TRUE(tile->cooperative());
    EXPECT_EQ(tile->required_space(), tiles::Space::Shared);
    EXPECT_TRUE(tile->reads());
}

// A 2D block tile of a row-major matrix (all-sequential nest): the source is a
// two-mode layout with the tile extents and the matrix strides.
TEST(TileAnalysisTest, Box2D_MultiDimSource) {
    builder::StructuredSDFGBuilder builder("ta_2dbox", FunctionType_CPU);
    types::Scalar sym(types::PrimitiveType::UInt64);
    builder.add_container("M", sym, true);
    builder.add_container("N", sym, true);
    builder.add_container("i_tile", sym);
    builder.add_container("j_tile", sym);
    builder.add_container("i", sym);
    builder.add_container("j", sym);
    types::Scalar elem(types::PrimitiveType::Double);
    builder.add_container("C", elem);
    types::Pointer ptr(elem);
    builder.add_container("A", ptr, true);

    auto MC = symbolic::integer(64);
    auto NC = symbolic::integer(32);
    auto M = symbolic::symbol("M");
    auto N = symbolic::symbol("N");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, M), symbolic::integer(0), symbolic::add(i_tile, MC));
    auto& j_tile_loop =
        builder
            .add_for(i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, N), symbolic::integer(0), symbolic::add(j_tile, NC));
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, MC)), symbolic::Lt(i, M)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, NC)), symbolic::Lt(j, N)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    auto& block = builder.add_block(j_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tk, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, tk, "_in2", {symbolic::add(symbolic::mul(i, N), j)}, ptr);
    builder.add_computational_memlet(block, tk, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    auto& ta = am.get<tiles::TileAnalysis>();
    const auto* tile = ta.tile(i_loop, "A");
    ASSERT_NE(tile, nullptr);
    EXPECT_TRUE(tile->axes().empty()); // all-sequential nest
    EXPECT_EQ(tile->required_space(), tiles::Space::Register);
    EXPECT_TRUE(tile->reads());
    // Colex source: inner j-mode (extent 32, stride 1), outer i-mode (extent 64, stride N).
    auto src = tiles::coalesce(tile->source());
    ASSERT_EQ(src.rank(), 2u);
    EXPECT_TRUE(symbolic::eq(src.shape()[0], symbolic::integer(32)));
    EXPECT_TRUE(symbolic::eq(src.stride()[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(src.shape()[1], symbolic::integer(64)));
    EXPECT_TRUE(symbolic::eq(src.stride()[1], N));
    EXPECT_TRUE(symbolic::eq(tile->source().offset(), symbolic::add(symbolic::mul(N, i_tile), j_tile)));
}

// Strided column of a 2D array + write direction: source has the row stride, and
// the tile is write-only.
TEST(TileAnalysisTest, StridedColumn_WriteDirection) {
    builder::StructuredSDFGBuilder builder("ta_column", FunctionType_CPU);
    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Array row(elem, symbolic::integer(8)); // float[8]
    types::Pointer ptr(row); // float (*)[8]
    types::Pointer flat_ptr(elem);
    types::Pointer opaque;
    builder.add_container("i", sym);
    builder.add_container("A", flat_ptr, true);
    builder.add_container("C", ptr, true);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tk, "_in", {i}, flat_ptr);
    builder.add_computational_memlet(block, tk, "_out", c_out, {i, symbolic::integer(3)}, ptr);

    analysis::AnalysisManager am(builder.subject());
    auto& ta = am.get<tiles::TileAnalysis>();
    const auto* tile = ta.tile(loop, "C");
    ASSERT_NE(tile, nullptr);
    EXPECT_TRUE(tile->writes());
    EXPECT_FALSE(tile->reads());
    EXPECT_TRUE(tile->axes().empty());
    // Column is strided: coalesced source is extent-4, stride-8 (row length), at col 3.
    auto src = tiles::coalesce(tile->source());
    ASSERT_EQ(src.rank(), 1u);
    EXPECT_TRUE(symbolic::eq(src.shape()[0], symbolic::integer(4)));
    EXPECT_TRUE(symbolic::eq(src.stride()[0], symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(tile->source().offset(), symbolic::integer(3)));
}

// A tiled 1D stencil: the over-approximated bounding box includes the halo, so the
// source extent is the tile width plus the radius on both sides.
TEST(TileAnalysisTest, Halo1D_OverApproxSource) {
    builder::StructuredSDFGBuilder builder("ta_halo", FunctionType_CPU);
    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("N", sym, true);
    builder.add_container("i_tile", sym);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);

    auto IT = symbolic::integer(8);
    auto N = symbolic::symbol("N");
    auto i_tile = symbolic::symbol("i_tile");
    auto i = symbolic::symbol("i");
    auto& root = builder.subject().root();
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, IT));
    auto& i_loop = builder.add_for(
        i_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, IT)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );
    auto& block = builder.add_block(i_loop.root());
    auto& a_lo = builder.add_access(block, "A");
    auto& a_hi = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    // A[i-1] + A[i+1]: bounding box [i_tile-1, i_tile+IT] -> extent IT + 2.
    builder.add_computational_memlet(block, a_lo, tk, "_in1", {symbolic::sub(i, symbolic::one())}, ptr);
    builder.add_computational_memlet(block, a_hi, tk, "_in2", {symbolic::add(i, symbolic::one())}, ptr);
    builder.add_computational_memlet(block, tk, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    auto& ta = am.get<tiles::TileAnalysis>();
    const auto* tile = ta.tile(i_loop, "A");
    ASSERT_NE(tile, nullptr);
    EXPECT_TRUE(tile->reads());
    auto src = tiles::coalesce(tile->source());
    ASSERT_EQ(src.rank(), 1u);
    EXPECT_TRUE(symbolic::eq(src.shape()[0], symbolic::integer(10))); // IT(8) + 2*radius(1)
    EXPECT_TRUE(symbolic::eq(src.stride()[0], symbolic::integer(1)));
    EXPECT_TRUE(symbolic::eq(tile->source().offset(), symbolic::sub(i_tile, symbolic::one())));
}

// OpenMP (CPU-parallel) map, tile shared across threads -> cooperation cannot use a
// private stack, so the required space is Global.
TEST(TileAnalysisTest, CpuParallel_CooperativeGlobal) {
    builder::StructuredSDFGBuilder builder("ta_omp_coop", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque;
    builder.add_container("i", sym);
    builder.add_container("t", sym);
    builder.add_container("N", sym, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", opaque, true);

    auto i = symbolic::symbol("i");
    auto t = symbolic::symbol("t");
    auto N = symbolic::symbol("N");
    auto& map_i = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        omp::ScheduleType_OMP::create()
    );
    auto& loop_t = builder.add_for(
        map_i.root(),
        t,
        symbolic::Lt(t, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(t, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop_t.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tk, "_in", {t}, ptr); // A[t]: i not in base -> cooperative
    builder.add_computational_memlet(block, tk, "_out", c_out, {t}, ptr);

    analysis::AnalysisManager am(builder.subject());
    auto& ta = am.get<tiles::TileAnalysis>();
    const auto* tile = ta.tile(loop_t, "A");
    ASSERT_NE(tile, nullptr);
    ASSERT_EQ(tile->axes().size(), 1u);
    EXPECT_FALSE(tile->axes()[0].schedule().has_scratchpad());
    EXPECT_EQ(tile->axes()[0].role(), tiles::Role::Cooperative);
    EXPECT_EQ(tile->axes()[0].schedule().level(), tiles::Level::Device);
    EXPECT_TRUE(tile->cooperative());
    EXPECT_EQ(tile->required_space(), tiles::Space::Global);
}

// OpenMP map whose indvar addresses the base -> per-thread private (registers).
TEST(TileAnalysisTest, CpuParallel_PerThreadRegisters) {
    builder::StructuredSDFGBuilder builder("ta_omp_perthread", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque;
    builder.add_container("i", sym);
    builder.add_container("t", sym);
    builder.add_container("N", sym, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", opaque, true);

    auto i = symbolic::symbol("i");
    auto t = symbolic::symbol("t");
    auto N = symbolic::symbol("N");
    auto& map_i = builder.add_map(
        root,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        omp::ScheduleType_OMP::create()
    );
    auto& loop_t = builder.add_for(
        map_i.root(),
        t,
        symbolic::Lt(t, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(t, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop_t.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto idx = symbolic::add(symbolic::mul(i, symbolic::integer(8)), t); // A[i*8+t]: i in base -> private
    builder.add_computational_memlet(block, a_in, tk, "_in", {idx}, ptr);
    builder.add_computational_memlet(block, tk, "_out", c_out, {t}, ptr);

    analysis::AnalysisManager am(builder.subject());
    auto& ta = am.get<tiles::TileAnalysis>();
    const auto* tile = ta.tile(loop_t, "A");
    ASSERT_NE(tile, nullptr);
    ASSERT_EQ(tile->axes().size(), 1u);
    EXPECT_FALSE(tile->axes()[0].schedule().has_scratchpad());
    EXPECT_EQ(tile->axes()[0].role(), tiles::Role::Private);
    EXPECT_FALSE(tile->cooperative());
    EXPECT_EQ(tile->required_space(), tiles::Space::Register);
}

// =====================================================================
// summarize(): read / write / alias classification from the dataflow
// =====================================================================

TEST(TileAnalysisTest, Summarize_ReadOnly) {
    builder::StructuredSDFGBuilder builder("ta_read_only", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {i}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem);

    auto summary = tiles::TileAnalysis::summarize(builder.subject(), loop, "A");
    EXPECT_TRUE(summary.reads);
    EXPECT_FALSE(summary.writes);
    EXPECT_FALSE(summary.aliased);
}

TEST(TileAnalysisTest, Summarize_WriteOnly) {
    builder::StructuredSDFGBuilder builder("ta_write_only", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& c_in = builder.add_access(block, "C");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in", {}, elem);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {i}, ptr);

    auto summary = tiles::TileAnalysis::summarize(builder.subject(), loop, "A");
    EXPECT_FALSE(summary.reads);
    EXPECT_TRUE(summary.writes);
    EXPECT_FALSE(summary.aliased);
}

TEST(TileAnalysisTest, Summarize_ReadAndWrite) {
    builder::StructuredSDFGBuilder builder("ta_read_write", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );
    // A[i] = A[i] + C
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {i}, ptr);
    builder.add_computational_memlet(block, c_in, tasklet, "_in2", {}, elem);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {i}, ptr);

    auto summary = tiles::TileAnalysis::summarize(builder.subject(), loop, "A");
    EXPECT_TRUE(summary.reads);
    EXPECT_TRUE(summary.writes);
    EXPECT_FALSE(summary.aliased);
}

TEST(TileAnalysisTest, Summarize_Unused) {
    builder::StructuredSDFGBuilder builder("ta_unused", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true); // never accessed in the loop
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in", {}, elem);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem);

    auto summary = tiles::TileAnalysis::summarize(builder.subject(), loop, "A");
    EXPECT_FALSE(summary.reads);
    EXPECT_FALSE(summary.writes);
    EXPECT_FALSE(summary.aliased);
}
