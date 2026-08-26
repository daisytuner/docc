#include "sdfg/transformations/local_storage.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/metadata_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;
using transformations::LocalStorage;

namespace {

/// Indices of tile dimensions whose extent is not 1 (i.e. real buffer dims).
std::vector<size_t> varying_dims(const analysis::MemoryTile& tile) {
    std::vector<size_t> dims;
    auto extents = tile.extents_approx();
    for (size_t d = 0; d < extents.size(); d++) {
        if (!extents[d].is_null() && !symbolic::eq(extents[d], symbolic::integer(1))) {
            dims.push_back(d);
        }
    }
    return dims;
}

/// True if every tile extent resolves to an integer constant (CPU precondition).
bool all_extents_integer(const analysis::MemoryTile& tile) {
    for (auto& e : tile.extents_approx()) {
        if (e.is_null() || !SymEngine::is_a<SymEngine::Integer>(*e)) {
            return false;
        }
    }
    return true;
}

/// Tile extents (as a copy) for concise assertions.
symbolic::MultiExpression extents_of(const analysis::MemoryTileGroup& group) { return group.tile.extents_approx(); }
/// Tile layout strides (as a vector) for concise assertions.
std::vector<symbolic::Expression> strides_of(const analysis::MemoryTileGroup& group) {
    return {group.tile.layout.strides().begin(), group.tile.layout.strides().end()};
}

} // namespace

// =====================================================================
// IsRead / IsWrite classification
// =====================================================================

TEST(LocalStorageTest, IsRead_ReadOnly) {
    builder::StructuredSDFGBuilder builder("ls_read_only", FunctionType_CPU);

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

    auto summary = LocalStorage::summarize(builder.subject(), loop, "A");
    EXPECT_TRUE(summary.reads);
    EXPECT_FALSE(summary.writes);
    EXPECT_FALSE(summary.aliased);
}

TEST(LocalStorageTest, IsWrite_WriteOnly) {
    builder::StructuredSDFGBuilder builder("ls_write_only", FunctionType_CPU);

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

    auto summary = LocalStorage::summarize(builder.subject(), loop, "A");
    EXPECT_FALSE(summary.reads);
    EXPECT_TRUE(summary.writes);
    EXPECT_FALSE(summary.aliased);
}

TEST(LocalStorageTest, IsReadWrite_ReadAndWrite) {
    builder::StructuredSDFGBuilder builder("ls_read_write", FunctionType_CPU);

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

    auto summary = LocalStorage::summarize(builder.subject(), loop, "A");
    EXPECT_TRUE(summary.reads);
    EXPECT_TRUE(summary.writes);
    EXPECT_FALSE(summary.aliased);
}

TEST(LocalStorageTest, IsReadWrite_Unused) {
    builder::StructuredSDFGBuilder builder("ls_unused", FunctionType_CPU);

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

    auto summary = LocalStorage::summarize(builder.subject(), loop, "A");
    EXPECT_FALSE(summary.reads);
    EXPECT_FALSE(summary.writes);
    EXPECT_FALSE(summary.aliased);
}

/**
 * Tile_Constant: pointer read at a constant location inside the loop.
 *   for i = 0..4: C += A[5]
 * Expected tile: scalar (no varying dims) — register promotion of a load.
 */
TEST(LocalStorageTest, Tile_Constant) {
    builder::StructuredSDFGBuilder builder("ls_cpu_constant", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque;
    builder.add_container("i", sym);
    builder.add_container("A", opaque, true);
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
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::integer(5)}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    auto summary = LocalStorage::summarize(builder.subject(), loop, "A");
    EXPECT_TRUE(summary.reads);
    EXPECT_FALSE(summary.writes);

    auto* group = LocalStorage::tile(loop, "A", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(all_extents_integer(group->tile));
    EXPECT_EQ(varying_dims(group->tile).size(), 0u); // scalar tile
}

/**
 * Tile_Accumulator: pointer used as an accumulator at a constant location.
 *   for i = 0..4: C[0] = C[0] + A[i]
 * Expected tile: scalar (no varying dims), read AND written.
 */
TEST(LocalStorageTest, Tile_Accumulator) {
    builder::StructuredSDFGBuilder builder("ls_cpu_accumulator", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque;
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", opaque, true);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );
    // C[0] = C[0] + A[i]
    auto& block = builder.add_block(loop.root());
    auto& c_in = builder.add_access(block, "C");
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {symbolic::integer(0)}, ptr);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::integer(0)}, ptr);

    analysis::AnalysisManager am(builder.subject());
    auto summary = LocalStorage::summarize(builder.subject(), loop, "C");
    EXPECT_TRUE(summary.reads);
    EXPECT_TRUE(summary.writes);

    auto* group = LocalStorage::tile(loop, "C", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(all_extents_integer(group->tile));
    EXPECT_EQ(varying_dims(group->tile).size(), 0u); // scalar accumulator tile
}

/**
 * Tile_Row: contiguous row of a linearized 2D array, localized at the
 * (single, outermost) loop.
 *   for j = 0..8: C[2*8 + j] = A[j]
 * Row = fixed row 2, varying column j → dense stride-1 tile of extent 8.
 */
TEST(LocalStorageTest, Tile_Row) {
    builder::StructuredSDFGBuilder builder("ls_cpu_tile_row", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque;
    builder.add_container("j", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", opaque, true);

    auto j = symbolic::symbol("j");
    auto& loop = builder.add_for(
        builder.subject().root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );
    // C[2*8 + j] = A[j]
    auto row_idx = symbolic::add(symbolic::mul(symbolic::integer(2), symbolic::integer(8)), j);
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {j}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {row_idx}, ptr);

    analysis::AnalysisManager am(builder.subject());
    EXPECT_TRUE(LocalStorage::summarize(builder.subject(), loop, "C").writes);

    auto* group = LocalStorage::tile(loop, "C", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(all_extents_integer(group->tile));

    auto dims = varying_dims(group->tile);
    ASSERT_EQ(dims.size(), 1u);
    auto extents = group->tile.extents_approx();
    EXPECT_TRUE(symbolic::eq(extents[dims[0]], symbolic::integer(8)));

    // Row is contiguous: the varying dimension has unit stride.
    std::vector<symbolic::Expression> strides(group->tile.layout.strides().begin(), group->tile.layout.strides().end());
    EXPECT_TRUE(symbolic::eq(strides[dims[0]], symbolic::integer(1)));
}

/**
 * Tile_Column: strided column of a 2D array, localized at the (single,
 * outermost) loop.
 *   float (*C)[8]; for i = 0..4: C[i][3] = A[i]
 * Column = fixed column 3, varying row i → dense buffer of extent 4 packing a
 * stride-8 access.
 *
 * Unlike a contiguous row (recoverable from a flat linearized index), a strided
 * column is only tileable when the row width is known from the layout — hence
 * the 2D pointer type. A single flat access such as C[i*8+3] carries no
 * dimension bounded by 8, so the width cannot be inferred and no tile forms.
 */
TEST(LocalStorageTest, Tile_Column) {
    builder::StructuredSDFGBuilder builder("ls_cpu_tile_column", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Array row(elem, symbolic::integer(8)); // float[8]
    types::Pointer ptr(row); // float (*)[8]
    types::Pointer flat_ptr(elem);
    types::Pointer opaque;
    builder.add_container("i", sym);
    builder.add_container("A", flat_ptr, true);
    builder.add_container("C", opaque, true);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );
    // C[i][3] = A[i]
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {i}, flat_ptr);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {i, symbolic::integer(3)}, ptr);

    analysis::AnalysisManager am(builder.subject());
    EXPECT_TRUE(LocalStorage::summarize(builder.subject(), loop, "C").writes);

    auto* group = LocalStorage::tile(loop, "C", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(all_extents_integer(group->tile));

    auto dims = varying_dims(group->tile);
    ASSERT_EQ(dims.size(), 1u);
    auto extents = group->tile.extents_approx();
    EXPECT_TRUE(symbolic::eq(extents[dims[0]], symbolic::integer(4)));

    // Column is strided: the varying (row) dimension has stride 8 (row length).
    std::vector<symbolic::Expression> strides(group->tile.layout.strides().begin(), group->tile.layout.strides().end());
    EXPECT_TRUE(symbolic::eq(strides[dims[0]], symbolic::integer(8)));
}

/**
 * Tile_2DBox: block-wise consumer of a 2D matrix.
 *
 *   for i_tile = 0..M step MC:
 *       for j_tile = 0..N step NC:
 *           for i = i_tile..min(i_tile+MC, M):     // localize here
 *               for j = j_tile..min(j_tile+NC, N):
 *                   C += A[i*N + j]
 *
 * At the i loop the tile is a MC x NC block of A. The extents are integer
 * constants (the tile sizes) even though the strides remain symbolic (row
 * length N) — exactly what a block-wise matrix consumer wants.
 */
TEST(LocalStorageTest, Tile_2DBox) {
    builder::StructuredSDFGBuilder builder("ls_cpu_2dbox", FunctionType_CPU);

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

    // C += A[i*N + j]
    auto& block = builder.add_block(j_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::add(symbolic::mul(i, N), j)}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    auto* group = LocalStorage::tile(i_loop, "A", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(LocalStorage::is_constant_bounded(group));

    auto dims = varying_dims(group->tile);
    ASSERT_EQ(dims.size(), 2u);
    auto extents = extents_of(*group);
    EXPECT_TRUE(symbolic::eq(extents[dims[0]], MC)); // 64 rows
    EXPECT_TRUE(symbolic::eq(extents[dims[1]], NC)); // 32 cols

    // Row-major block of a matrix with symbolic row length N: outer stride N,
    // inner stride 1.
    auto strides = strides_of(*group);
    EXPECT_TRUE(symbolic::eq(strides[dims[0]], N));
    EXPECT_TRUE(symbolic::eq(strides[dims[1]], symbolic::integer(1)));
}

namespace {

/// Builds a tiled 2D r-point stencil `B[i,j] = sum A[i±r, j], A[i, j±r]`
/// (linearized over row length M, tiled by IT x JT = 32 x 32) and returns the
/// inner i loop and the center A access node. The bounding-box tile at the i
/// loop must include the halo, giving extents (IT + 2r) x (JT + 2r).
struct TiledStencil {
    structured_control_flow::StructuredLoop* i_loop;
    data_flow::AccessNode* a_center;
};

TiledStencil build_tiled_stencil(builder::StructuredSDFGBuilder& builder, long radius) {
    types::Scalar sym(types::PrimitiveType::UInt64);
    builder.add_container("N", sym, true);
    builder.add_container("M", sym, true);
    builder.add_container("i_tile", sym);
    builder.add_container("j_tile", sym);
    builder.add_container("i", sym);
    builder.add_container("j", sym);

    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("A", ptr, true);
    builder.add_container("B", ptr, true);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto IT = symbolic::integer(32);
    auto JT = symbolic::integer(32);
    auto r = symbolic::integer(radius);

    auto N_lo = symbolic::sub(N, symbolic::integer(radius));
    auto M_lo = symbolic::sub(M, symbolic::integer(radius));

    auto& root = builder.subject().root();
    auto& i_tile_loop = builder.add_for(root, i_tile, symbolic::Lt(i_tile, N_lo), r, symbolic::add(i_tile, IT));
    auto& j_tile_loop =
        builder.add_for(i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, M_lo), r, symbolic::add(j_tile, JT));
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, IT)), symbolic::Lt(i, N_lo)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, JT)), symbolic::Lt(j, M_lo)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    auto center = symbolic::add(symbolic::mul(i, M), j);
    auto north = symbolic::add(symbolic::mul(symbolic::sub(i, r), M), j);
    auto south = symbolic::add(symbolic::mul(symbolic::add(i, r), M), j);
    auto west = symbolic::add(symbolic::mul(i, M), symbolic::sub(j, r));
    auto east = symbolic::add(symbolic::mul(i, M), symbolic::add(j, r));

    // B[center] = A[center]
    auto& block = builder.add_block(j_loop.root());
    auto& a_center = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& t0 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_center, t0, "_in", {center}, ptr);
    builder.add_computational_memlet(block, t0, "_out", b_out, {center}, ptr);

    auto add_neighbor = [&](const symbolic::Expression& off) {
        auto& blk = builder.add_block(j_loop.root());
        auto& a_n = builder.add_access(blk, "A");
        auto& b_in = builder.add_access(blk, "B");
        auto& b_o = builder.add_access(blk, "B");
        auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(blk, b_in, t, "_in1", {center}, ptr);
        builder.add_computational_memlet(blk, a_n, t, "_in2", {off}, ptr);
        builder.add_computational_memlet(blk, t, "_out", b_o, {center}, ptr);
    };
    add_neighbor(north);
    add_neighbor(south);
    add_neighbor(west);
    add_neighbor(east);

    return {&i_loop, &a_center};
}

} // namespace

/**
 * Tile_2DBoxWithHalo_Radius1: 5-point stencil (radius 1). The tile at the i
 * loop must include the +/-1 halo in each dimension → extents 34 x 34.
 */
TEST(LocalStorageTest, Tile_2DBoxWithHalo_Radius1) {
    builder::StructuredSDFGBuilder builder("ls_cpu_halo_r1", FunctionType_CPU);
    auto fix = build_tiled_stencil(builder, /*radius=*/1);

    analysis::AnalysisManager am(builder.subject());
    auto* group = LocalStorage::tile(*fix.i_loop, "A", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(LocalStorage::is_constant_bounded(group));

    auto dims = varying_dims(group->tile);
    ASSERT_EQ(dims.size(), 2u);
    auto extents = extents_of(*group);
    EXPECT_TRUE(symbolic::eq(extents[dims[0]], symbolic::integer(34))); // IT + 2*1
    EXPECT_TRUE(symbolic::eq(extents[dims[1]], symbolic::integer(34))); // JT + 2*1
}

/**
 * Tile_2DBoxWithHalo_Radius2: 5-point stencil (radius 2). The halo grows to
 * +/-2 in each dimension → extents 36 x 36.
 */
TEST(LocalStorageTest, Tile_2DBoxWithHalo_Radius2) {
    builder::StructuredSDFGBuilder builder("ls_cpu_halo_r2", FunctionType_CPU);
    auto fix = build_tiled_stencil(builder, /*radius=*/2);

    analysis::AnalysisManager am(builder.subject());
    auto* group = LocalStorage::tile(*fix.i_loop, "A", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(LocalStorage::is_constant_bounded(group));

    auto dims = varying_dims(group->tile);
    ASSERT_EQ(dims.size(), 2u);
    auto extents = extents_of(*group);
    EXPECT_TRUE(symbolic::eq(extents[dims[0]], symbolic::integer(36))); // IT + 2*2
    EXPECT_TRUE(symbolic::eq(extents[dims[1]], symbolic::integer(36))); // JT + 2*2
}

// =====================================================================
// Negative cases: no constantly bounded tile
// =====================================================================

/**
 * Tile_Negative_SymbolicBound: an untiled loop over a symbolic bound N produces a
 * tile whose extent is N — a coherent tile, but NOT a compile-time constant, so
 * a CPU local-storage transformation must reject it.
 */
TEST(LocalStorageTest, Tile_Negative_SymbolicBound) {
    builder::StructuredSDFGBuilder builder("ls_neg_symbolic", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("N", sym, true);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(), i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one())
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    auto* group = LocalStorage::tile(loop, "A", am);
    ASSERT_NE(group, nullptr); // a tile exists...
    EXPECT_FALSE(LocalStorage::is_constant_bounded(group)); // ...but its extent is symbolic (N)
}

/**
 * Tile_Negative_IndirectGather: a data-dependent (gather) index A[p], where p is
 * loaded from memory inside the loop, is not affine and cannot be bounded, so no
 * constantly bounded tile exists.
 */
TEST(LocalStorageTest, Tile_Negative_IndirectGather) {
    builder::StructuredSDFGBuilder builder("ls_neg_gather", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer fptr(elem);
    types::Pointer iptr(sym);
    builder.add_container("i", sym);
    builder.add_container("p", sym); // data-dependent index
    builder.add_container("idx", iptr, true);
    builder.add_container("A", fptr, true);
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto p = symbolic::symbol("p");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );

    // p = idx[i]
    auto& b1 = builder.add_block(loop.root());
    auto& idx_in = builder.add_access(b1, "idx");
    auto& p_out = builder.add_access(b1, "p");
    auto& t1 = builder.add_tasklet(b1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(b1, idx_in, t1, "_in", {i}, iptr);
    builder.add_computational_memlet(b1, t1, "_out", p_out, {}, sym);

    // C += A[p]
    auto& b2 = builder.add_block(loop.root());
    auto& a_in = builder.add_access(b2, "A");
    auto& c_in = builder.add_access(b2, "C");
    auto& c_out = builder.add_access(b2, "C");
    auto& t2 = builder.add_tasklet(b2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(b2, c_in, t2, "_in1", {}, elem);
    builder.add_computational_memlet(b2, a_in, t2, "_in2", {p}, fptr);
    builder.add_computational_memlet(b2, t2, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    auto* group = LocalStorage::tile(loop, "A", am);
    EXPECT_FALSE(LocalStorage::is_constant_bounded(group)); // no analyzable, constant tile
}

/**
 * Tile_Degenerate_ConstantButHuge: two far-apart constant indices A[0], A[100] merge
 * into one group (constant base difference). The bounding box spans [0, 100], so
 * the tile is a CONSTANT 101-element buffer for only 2 accessed elements.
 *
 * MLA reports a large integer extent (NOT null) — overapproximate() only nulls
 * out sentinel-dependent extents, never constant ones. So is_constant_bounded is
 * TRUE here: it is necessary but NOT sufficient. It cannot see the degenerate
 * (sparse) density; a separate size/density guard is required.
 */
TEST(LocalStorageTest, Tile_Degenerate_ConstantButHuge) {
    builder::StructuredSDFGBuilder builder("ls_degenerate", FunctionType_CPU);

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
        symbolic::add(i, symbolic::one())
    );
    // C += A[0] + A[100]
    auto& block = builder.add_block(loop.root());
    auto& a_lo = builder.add_access(block, "A");
    auto& a_hi = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_lo, t, "_in1", {symbolic::integer(0)}, ptr);
    builder.add_computational_memlet(block, a_hi, t, "_in2", {symbolic::integer(100)}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);
    // keep C an accumulator so the block is well-formed
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);

    analysis::AnalysisManager am(builder.subject());
    auto* group = LocalStorage::tile(loop, "A", am);
    ASSERT_NE(group, nullptr);

    // The extent is a large constant, not null.
    auto extents = extents_of(*group);
    ASSERT_EQ(extents.size(), 1u);
    ASSERT_FALSE(extents[0].is_null());
    EXPECT_TRUE(symbolic::eq(extents[0], symbolic::integer(101)));

    // is_constant_bounded passes despite the buffer being 101 slots for 2 uses.
    EXPECT_TRUE(LocalStorage::is_constant_bounded(group));
}

/**
 * Tile_MultipleGroups_SymbolicBases: SYR2K-style A[i*8+k] and A[j*8+k] read at the k
 * loop. Their bases (i*8 vs j*8) differ symbolically, so they do NOT merge — the
 * container has TWO tile groups at this scope. Each is on its own constant-bounded
 * (extent 8), but the container is NOT localizable as a single tile in v1.
 */
TEST(LocalStorageTest, Tile_MultipleGroups_SymbolicBases) {
    builder::StructuredSDFGBuilder builder("ls_multigroup", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("j", sym);
    builder.add_container("k", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto K = symbolic::integer(8);

    auto& root = builder.subject().root();
    auto& i_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::one())
    );
    auto& j_loop = builder.add_for(
        i_loop.root(), j, symbolic::Lt(j, symbolic::integer(4)), symbolic::integer(0), symbolic::add(j, symbolic::one())
    );
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // C += A[i*8+k]
    auto& b1 = builder.add_block(k_loop.root());
    auto& a_ik = builder.add_access(b1, "A");
    auto& c_in1 = builder.add_access(b1, "C");
    auto& c_out1 = builder.add_access(b1, "C");
    auto& t1 = builder.add_tasklet(b1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(b1, c_in1, t1, "_in1", {}, elem);
    builder.add_computational_memlet(b1, a_ik, t1, "_in2", {symbolic::add(symbolic::mul(i, K), k)}, ptr);
    builder.add_computational_memlet(b1, t1, "_out", c_out1, {}, elem);

    // C += A[j*8+k]
    auto& b2 = builder.add_block(k_loop.root());
    auto& a_jk = builder.add_access(b2, "A");
    auto& c_in2 = builder.add_access(b2, "C");
    auto& c_out2 = builder.add_access(b2, "C");
    auto& t2 = builder.add_tasklet(b2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(b2, c_in2, t2, "_in1", {}, elem);
    builder.add_computational_memlet(b2, a_jk, t2, "_in2", {symbolic::add(symbolic::mul(j, K), k)}, ptr);
    builder.add_computational_memlet(b2, t2, "_out", c_out2, {}, elem);

    analysis::AnalysisManager am(builder.subject());

    // Two distinct groups at the k loop (bases i*8 vs j*8 do not merge).
    auto* groups = LocalStorage::tile_groups(k_loop, "A", am);
    ASSERT_NE(groups, nullptr);
    ASSERT_EQ(groups->size(), 2u);

    // Each group is a 2D tile [i..i] x [0..7] (resp. j) — one varying dim of 8.
    for (const auto& g : *groups) {
        EXPECT_TRUE(LocalStorage::is_constant_bounded(&g));
        auto dims = varying_dims(g.tile);
        ASSERT_EQ(dims.size(), 1u);
        EXPECT_TRUE(symbolic::eq(extents_of(g)[dims[0]], K));
    }

    // Container-anchored: a multi-group container is NOT localizable in v1.
    EXPECT_EQ(LocalStorage::tile(k_loop, "A", am), nullptr);
}

// =====================================================================
// Localizability gate: single-group coherence (tile)
// =====================================================================

/**
 * Tile_SingleGroup_Localizable: a container whose every access forms one coherent
 * tile is localizable — tile returns that group.
 */
TEST(LocalStorageTest, Tile_SingleGroup_Localizable) {
    builder::StructuredSDFGBuilder builder("ls_single_group", FunctionType_CPU);

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
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    // C += A[i] (single contiguous group)
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    auto* groups = LocalStorage::tile_groups(loop, "A", am);
    ASSERT_NE(groups, nullptr);
    EXPECT_EQ(groups->size(), 1u);

    auto* single = LocalStorage::tile(loop, "A", am);
    ASSERT_NE(single, nullptr);
    EXPECT_EQ(single, &groups->front());
}

/**
 * Tile_SplitNode_NotLocalizable: a SINGLE access node with two out-edges landing
 * in two different groups (A[i*8+k] and A[j*8+k]) is a split node. The container
 * has two groups → not localizable, even though there is only one access node.
 */
TEST(LocalStorageTest, Tile_SplitNode_NotLocalizable) {
    builder::StructuredSDFGBuilder builder("ls_split_node", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("j", sym);
    builder.add_container("k", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto K = symbolic::integer(8);

    auto& root = builder.subject().root();
    auto& i_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::one())
    );
    auto& j_loop = builder.add_for(
        i_loop.root(), j, symbolic::Lt(j, symbolic::integer(4)), symbolic::integer(0), symbolic::add(j, symbolic::one())
    );
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // tmp = A[i*8+k] + A[j*8+k], both edges from the SAME access node.
    auto& block = builder.add_block(k_loop.root());
    auto& a_node = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_node, t, "_in1", {symbolic::add(symbolic::mul(i, K), k)}, ptr);
    builder.add_computational_memlet(block, a_node, t, "_in2", {symbolic::add(symbolic::mul(j, K), k)}, ptr);
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    EXPECT_EQ(LocalStorage::tile_groups(k_loop, "A", am)->size(), 2u);
    EXPECT_EQ(LocalStorage::tile(k_loop, "A", am), nullptr);
}

/**
 * Tile_UngroupedMemlet_NotLocalizable: a container mixing an analyzable access
 * A[i] with an unanalyzable (gather) access A[p]. MLA drops the container's tile
 * entirely (no groups), so tile rejects — a container is only localizable
 * when ALL of its accesses are analyzable.
 */
TEST(LocalStorageTest, Tile_UngroupedMemlet_NotLocalizable) {
    builder::StructuredSDFGBuilder builder("ls_ungrouped", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer fptr(elem);
    types::Pointer iptr(sym);
    builder.add_container("i", sym);
    builder.add_container("p", sym);
    builder.add_container("idx", iptr, true);
    builder.add_container("A", fptr, true);
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto p = symbolic::symbol("p");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );

    // block1: C += A[i] (analyzable, one group)
    auto& b1 = builder.add_block(loop.root());
    auto& a_aff = builder.add_access(b1, "A");
    auto& c_in1 = builder.add_access(b1, "C");
    auto& c_out1 = builder.add_access(b1, "C");
    auto& t1 = builder.add_tasklet(b1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(b1, c_in1, t1, "_in1", {}, elem);
    builder.add_computational_memlet(b1, a_aff, t1, "_in2", {i}, fptr);
    builder.add_computational_memlet(b1, t1, "_out", c_out1, {}, elem);

    // block2: p = idx[i]
    auto& b2 = builder.add_block(loop.root());
    auto& idx_in = builder.add_access(b2, "idx");
    auto& p_out = builder.add_access(b2, "p");
    auto& t2 = builder.add_tasklet(b2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(b2, idx_in, t2, "_in", {i}, iptr);
    builder.add_computational_memlet(b2, t2, "_out", p_out, {}, sym);

    // block3: C += A[p] (gather, ungrouped)
    auto& b3 = builder.add_block(loop.root());
    auto& a_gather = builder.add_access(b3, "A");
    auto& c_in3 = builder.add_access(b3, "C");
    auto& c_out3 = builder.add_access(b3, "C");
    auto& t3 = builder.add_tasklet(b3, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(b3, c_in3, t3, "_in1", {}, elem);
    builder.add_computational_memlet(b3, a_gather, t3, "_in2", {p}, fptr);
    builder.add_computational_memlet(b3, t3, "_out", c_out3, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    // MLA drops the whole container tile when any of its accesses is unanalyzable.
    EXPECT_EQ(LocalStorage::tile_groups(loop, "A", am), nullptr);
    EXPECT_EQ(LocalStorage::tile(loop, "A", am), nullptr);
}

// =====================================================================
// Size guard: buffer element count (product of extents)
// =====================================================================

/**
 * Tile_ElementCount_Dense: a contiguous 8-wide row packs to exactly 8 slots.
 */
TEST(LocalStorageTest, Tile_ElementCount_Dense) {
    builder::StructuredSDFGBuilder builder("ls_count_dense", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque;
    builder.add_container("j", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", opaque, true);

    auto j = symbolic::symbol("j");
    auto& loop = builder.add_for(
        builder.subject().root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::one())
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, t, "_in", {j}, ptr);
    builder.add_computational_memlet(
        block, t, "_out", c_out, {symbolic::add(symbolic::mul(symbolic::integer(2), symbolic::integer(8)), j)}, ptr
    );

    analysis::AnalysisManager am(builder.subject());
    auto* group = LocalStorage::tile(loop, "C", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(symbolic::eq(LocalStorage::tile_element_count(group), symbolic::integer(8)));
}

/**
 * Tile_ElementCount_Degenerate: the far-apart constant accesses A[0], A[100]
 * allocate 101 slots — the size guard can see the blowup that is_constant_bounded
 * cannot.
 */
TEST(LocalStorageTest, Tile_ElementCount_Degenerate) {
    builder::StructuredSDFGBuilder builder("ls_count_degenerate", FunctionType_CPU);

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
        symbolic::add(i, symbolic::one())
    );
    auto& block = builder.add_block(loop.root());
    auto& a_lo = builder.add_access(block, "A");
    auto& a_hi = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_lo, t, "_in1", {symbolic::integer(0)}, ptr);
    builder.add_computational_memlet(block, a_hi, t, "_in2", {symbolic::integer(100)}, ptr);
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    auto* group = LocalStorage::tile(loop, "A", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(LocalStorage::is_constant_bounded(group)); // passes the constant check
    EXPECT_TRUE(symbolic::eq(LocalStorage::tile_element_count(group), symbolic::integer(101))); // but 101 slots
}

// =====================================================================
// can_be_applied: legality, safety, capacity, and schedule derivation
// =====================================================================

/**
 * CanApply_InReadOnly_Accepts: a read-only pointer with a single constant tile is
 * accepted, and tile_info is populated.
 */
TEST(LocalStorageTest, CanApply_InReadOnly_Accepts) {
    builder::StructuredSDFGBuilder builder("ls_can_in_ok", FunctionType_CPU);

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
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, a_in);
    EXPECT_TRUE(xform.can_be_applied(builder, am));
    ASSERT_EQ(xform.tile_info().dimensions.size(), 1u);
    EXPECT_TRUE(symbolic::eq(xform.tile_info().dimensions[0], symbolic::integer(8)));
}

/**
 * CanApply_WriteOnly_Accepts: a written container is a valid localization target
 * (write-only → copy-out only); direction is derived, not gated.
 */
TEST(LocalStorageTest, CanApply_WriteOnly_Accepts) {
    builder::StructuredSDFGBuilder builder("ls_can_write_only", FunctionType_CPU);

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
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    // A[i] = C (A written)
    auto& block = builder.add_block(loop.root());
    auto& c_in = builder.add_access(block, "C");
    auto& a_out = builder.add_access(block, "A");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, c_in, t, "_in", {}, elem);
    builder.add_computational_memlet(block, t, "_out", a_out, {i}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, a_out);
    EXPECT_TRUE(xform.can_be_applied(builder, am));
}

/**
 * CanApply_OutWritten_Accepts: a written container is accepted.
 */
TEST(LocalStorageTest, CanApply_OutWritten_Accepts) {
    builder::StructuredSDFGBuilder builder("ls_can_out_ok", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    // C[i] = A[i] (C written)
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, t, "_in", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {i}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, c_out);
    EXPECT_TRUE(xform.can_be_applied(builder, am));
}

/**
 * CanApply_NonPointer_Rejects: a non-pointer (scalar) container is rejected.
 */
TEST(LocalStorageTest, CanApply_NonPointer_Rejects) {
    builder::StructuredSDFGBuilder builder("ls_can_nonptr", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem); // scalar

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, c_out); // C is a scalar, not a pointer
    EXPECT_FALSE(xform.can_be_applied(builder, am));
}

/**
 * CanApply_SymbolicBound_Rejects: a read-only tile whose extent is symbolic (N)
 * is not constant-bounded → rejected.
 */
TEST(LocalStorageTest, CanApply_SymbolicBound_Rejects) {
    builder::StructuredSDFGBuilder builder("ls_can_symbolic", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("N", sym, true);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(), i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one())
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, a_in);
    EXPECT_FALSE(xform.can_be_applied(builder, am));
}

/**
 * CanApply_MultiGroup_Rejects: a container with two symbolic-base groups is not a
 * single localizable tile → rejected.
 */
TEST(LocalStorageTest, CanApply_MultiGroup_Rejects) {
    builder::StructuredSDFGBuilder builder("ls_can_multi", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("j", sym);
    builder.add_container("k", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto K = symbolic::integer(8);

    auto& root = builder.subject().root();
    auto& i_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::one())
    );
    auto& j_loop = builder.add_for(
        i_loop.root(), j, symbolic::Lt(j, symbolic::integer(4)), symbolic::integer(0), symbolic::add(j, symbolic::one())
    );
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    auto& b1 = builder.add_block(k_loop.root());
    auto& a_ik = builder.add_access(b1, "A");
    auto& c_in1 = builder.add_access(b1, "C");
    auto& c_out1 = builder.add_access(b1, "C");
    auto& t1 = builder.add_tasklet(b1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(b1, c_in1, t1, "_in1", {}, elem);
    builder.add_computational_memlet(b1, a_ik, t1, "_in2", {symbolic::add(symbolic::mul(i, K), k)}, ptr);
    builder.add_computational_memlet(b1, t1, "_out", c_out1, {}, elem);

    auto& b2 = builder.add_block(k_loop.root());
    auto& a_jk = builder.add_access(b2, "A");
    auto& c_in2 = builder.add_access(b2, "C");
    auto& c_out2 = builder.add_access(b2, "C");
    auto& t2 = builder.add_tasklet(b2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(b2, c_in2, t2, "_in1", {}, elem);
    builder.add_computational_memlet(b2, a_jk, t2, "_in2", {symbolic::add(symbolic::mul(j, K), k)}, ptr);
    builder.add_computational_memlet(b2, t2, "_out", c_out2, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(k_loop, a_ik);
    EXPECT_FALSE(xform.can_be_applied(builder, am));
}

/**
 * CanApply_BudgetExceeded_Rejects: a constant but oversized tile (> 65536 slots)
 * is rejected by the capacity guard — distinct from the constant-bounded check,
 * which this tile passes.
 */
TEST(LocalStorageTest, CanApply_BudgetExceeded_Rejects) {
    builder::StructuredSDFGBuilder builder("ls_can_budget", FunctionType_CPU);

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
        symbolic::add(i, symbolic::one())
    );
    // C += A[0] + A[70000] → constant tile of 70001 slots, over the 65536 budget
    auto& block = builder.add_block(loop.root());
    auto& a_lo = builder.add_access(block, "A");
    auto& a_hi = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_lo, t, "_in1", {symbolic::integer(0)}, ptr);
    builder.add_computational_memlet(block, a_hi, t, "_in2", {symbolic::integer(70000)}, ptr);
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, a_lo);

    // The tile is a compile-time constant, but its slot count exceeds the budget.
    auto* group = LocalStorage::tile(loop, "A", am);
    ASSERT_NE(group, nullptr);
    EXPECT_TRUE(LocalStorage::is_constant_bounded(group));
    EXPECT_TRUE(symbolic::eq(LocalStorage::tile_element_count(group), symbolic::integer(70001)));
    EXPECT_FALSE(xform.can_be_applied(builder, am));
}

// =====================================================================
// Aliasing and side-effect safety (dataflow-based, no Users)
// =====================================================================

/**
 * CanApply_Aliased_Rejects: a reference memlet aliases the container into another
 * name. The alias can access the same memory outside our tracked memlets, so the
 * dataflow scan flags `aliased` and localization is rejected.
 */
TEST(LocalStorageTest, CanApply_Aliased_Rejects) {
    builder::StructuredSDFGBuilder builder("ls_alias", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("B", ptr, true); // alias target
    builder.add_container("C", elem);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    // C += A[i]
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    // B = &A[0] — reference memlet aliasing A
    auto& ref_block = builder.add_block(loop.root());
    auto& a_ref = builder.add_access(ref_block, "A");
    auto& b_ref = builder.add_access(ref_block, "B");
    builder.add_reference_memlet(ref_block, a_ref, b_ref, {symbolic::integer(0)}, ptr);

    EXPECT_TRUE(LocalStorage::summarize(builder.subject(), loop, "A").aliased);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, a_in);
    EXPECT_FALSE(xform.can_be_applied(builder, am));
}

/**
 * CanApply_SideEffect_Rejects: a side-effecting library node in the loop may
 * touch memory outside the tracked memlets, so localization is rejected.
 */
TEST(LocalStorageTest, CanApply_SideEffect_Rejects) {
    builder::StructuredSDFGBuilder builder("ls_side_effect", FunctionType_CPU);

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
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    // A genuine side-effecting library node (memset) in the loop body — unlike a
    // bare __syncthreads barrier, which accesses no data and does not block staging.
    auto& se_block = builder.add_block(loop.root());
    builder.add_library_node<stdlib::MemsetNode>(se_block, DebugInfo(), symbolic::integer(0), symbolic::integer(4));

    EXPECT_TRUE(LocalStorage::has_side_effect(loop));

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, a_in);
    EXPECT_FALSE(xform.can_be_applied(builder, am));
}

// A __syncthreads barrier (no data connectors) is not a staging-blocking side
// effect — needed so LocalStorage can stage a row across barrier-separated passes.
TEST(LocalStorageTest, HasSideEffect_BarrierIgnored) {
    builder::StructuredSDFGBuilder builder("ls_barrier_ok", FunctionType_CPU);
    types::Scalar sym(types::PrimitiveType::UInt64);
    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    auto& blk = builder.add_block(loop.root());
    builder.add_library_node<data_flow::BarrierLocalNode>(blk, DebugInfo());

    EXPECT_FALSE(LocalStorage::has_side_effect(loop));
}

/**
 * CanApply_LibraryNodeCaptures_Rejects: a library node consumes the container
 * pointer but has NO global side effect (side_effect()==false). The coarse
 * side_effect() flag misses it, but its pointer_access_type is unknown (null),
 * so the pointer analysis flags it as aliased — exactly the GEMM/swap gap.
 */
TEST(LocalStorageTest, CanApply_LibraryNodeCaptures_Rejects) {
    builder::StructuredSDFGBuilder builder("ls_libnode_capture", FunctionType_CPU);

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
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    // Normal read C += A[i]
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    // A library node that consumes A with unknown pointer semantics.
    auto& lib_block = builder.add_block(loop.root());
    auto& a_lib = builder.add_access(lib_block, "A");
    auto& lib = builder.add_library_node<data_flow::MetadataNode>(
        lib_block,
        DebugInfo(),
        std::vector<std::string>{},
        std::vector<std::string>{"_in"},
        std::unordered_map<std::string, std::string>{}
    );
    builder.add_computational_memlet(lib_block, a_lib, lib, "_in", {i}, ptr);

    // The library node has no global side effect...
    EXPECT_FALSE(LocalStorage::has_side_effect(loop));
    // ...yet the pointer analysis flags the capture.
    EXPECT_TRUE(LocalStorage::summarize(builder.subject(), loop, "A").aliased);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, a_in);
    EXPECT_FALSE(xform.can_be_applied(builder, am));
}

// =====================================================================
// Shared apply() (CPU path)
// =====================================================================

namespace {
/// True if any data node named @p name appears in @p block's dataflow.
bool block_uses(structured_control_flow::Block& block, const std::string& name) {
    for (auto* node : block.dataflow().data_nodes()) {
        if (node->data() == name) {
            return true;
        }
    }
    return false;
}

// The copy block may be wrapped in a boundary-guard IfElse (element predication
// of the global access). Unwrap it to reach the underlying copy Block.
structured_control_flow::Block* copy_block_of(structured_control_flow::ControlFlowNode& node) {
    if (auto* blk = dynamic_cast<structured_control_flow::Block*>(&node)) {
        return blk;
    }
    if (auto* ife = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        return dynamic_cast<structured_control_flow::Block*>(&ife->at(0).first.at(0));
    }
    return nullptr;
}
} // namespace

/**
 * Apply_In_Array: read-only cache. Emits a copy-in loop before the loop and
 * redirects the loop's reads to the buffer; no writeback.
 */
TEST(LocalStorageTest, Apply_In_Array) {
    builder::StructuredSDFGBuilder builder("ls_apply_in", FunctionType_CPU);

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
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {}, elem);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {}, elem);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, a_in);
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    xform.apply(builder, am);

    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));
    EXPECT_TRUE(builder.subject().type(buf) == types::Array(elem, symbolic::integer(8)));

    // Structure: [copy_in_loop, main_loop].
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 2u);
    auto* copy_loop = dyn_cast<structured_control_flow::Map*>(&root.at(0));
    ASSERT_NE(copy_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(copy_loop->condition(), symbolic::Lt(copy_loop->indvar(), symbolic::integer(8))));
    auto* main_loop = dyn_cast<structured_control_flow::For*>(&root.at(1));
    ASSERT_NE(main_loop, nullptr);

    // Main loop reads the buffer, not A.
    auto* main_block = dyn_cast<structured_control_flow::Block*>(&main_loop->root().at(0));
    ASSERT_NE(main_block, nullptr);
    EXPECT_TRUE(block_uses(*main_block, buf));
    EXPECT_FALSE(block_uses(*main_block, "A"));
}

/**
 * Apply_Out_WriteOnly: write-only accumulator. No copy-in (container is not
 * read), just a writeback loop after.
 */
TEST(LocalStorageTest, Apply_Out_WriteOnly) {
    builder::StructuredSDFGBuilder builder("ls_apply_out_wo", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    // C[i] = A[i]
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, t, "_in", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {i}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, c_out);
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    xform.apply(builder, am);

    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));

    // Structure: [main_loop, writeback_loop] — no copy-in.
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 2u);
    auto* main_loop = dyn_cast<structured_control_flow::For*>(&root.at(0));
    ASSERT_NE(main_loop, nullptr);
    auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&root.at(1));
    ASSERT_NE(wb_loop, nullptr);

    // Writeback reads the buffer and writes C (copy may be boundary-guarded).
    auto* wb_block = copy_block_of(wb_loop->root().at(0));
    ASSERT_NE(wb_block, nullptr);
    EXPECT_TRUE(block_uses(*wb_block, buf));
    EXPECT_TRUE(block_uses(*wb_block, "C"));
}

/**
 * Apply_Out_ReadWrite: read-write accumulator. Copy-in before, writeback after.
 */
TEST(LocalStorageTest, Apply_Out_ReadWrite) {
    builder::StructuredSDFGBuilder builder("ls_apply_out_rw", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    // C[i] = C[i] + A[i]
    auto& block = builder.add_block(loop.root());
    auto& c_in = builder.add_access(block, "C");
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {i}, ptr);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {i}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, c_out);
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    xform.apply(builder, am);

    auto buf = xform.local_container();

    // Structure: [copy_in_loop, main_loop, writeback_loop].
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 3u);
    EXPECT_NE(dyn_cast<structured_control_flow::Map*>(&root.at(0)), nullptr);
    EXPECT_NE(dyn_cast<structured_control_flow::For*>(&root.at(1)), nullptr);
    EXPECT_NE(dyn_cast<structured_control_flow::Map*>(&root.at(2)), nullptr);

    auto* main_loop = dyn_cast<structured_control_flow::For*>(&root.at(1));
    ASSERT_NE(main_loop, nullptr);
    auto* main_block = dyn_cast<structured_control_flow::Block*>(&main_loop->root().at(0));
    ASSERT_NE(main_block, nullptr);
    EXPECT_TRUE(block_uses(*main_block, buf));
    EXPECT_FALSE(block_uses(*main_block, "C"));
}

/**
 * Apply_Scalar_Accumulator: a scalar (extent-1) tile promotes to a 1-element
 * buffer with no copy loops — a plain copy block before and writeback after.
 */
TEST(LocalStorageTest, Apply_Scalar_Accumulator) {
    builder::StructuredSDFGBuilder builder("ls_apply_scalar", FunctionType_CPU);

    types::Scalar sym(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("i", sym);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);

    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::one())
    );
    // C[0] = C[0] + A[i]
    auto& block = builder.add_block(loop.root());
    auto& c_in = builder.add_access(block, "C");
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {symbolic::integer(0)}, ptr);
    builder.add_computational_memlet(block, a_in, t, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {symbolic::integer(0)}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop, c_out);
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    xform.apply(builder, am);

    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));
    EXPECT_TRUE(builder.subject().type(buf) == types::Array(elem, symbolic::integer(1)));

    // Scalar tile → copy-in and writeback are plain Blocks (no Map loops).
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 3u);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&root.at(0)), nullptr);
    EXPECT_NE(dyn_cast<structured_control_flow::For*>(&root.at(1)), nullptr);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&root.at(2)), nullptr);
}

// =====================================================================
// LocalityPlan: schedule classification (build_locality_plan)
// =====================================================================

namespace {

/// A cooperative/per-thread GPU dim at @p level for direct derive_storage tests.
LocalStorage::LocalityPlan::Dim
make_gpu_dim(bool cooperative, LocalStorage::LocalityPlan::Level level = LocalStorage::LocalityPlan::Level::Block) {
    LocalStorage::LocalityPlan::Dim d;
    d.indvar = symbolic::symbol("i");
    d.is_gpu = true;
    d.cooperative = cooperative;
    d.level = level;
    d.parallel_size = symbolic::integer(32);
    return d;
}

/// A cooperative/per-thread CPU parallel dim for direct derive_storage tests.
LocalStorage::LocalityPlan::Dim make_cpu_dim(bool cooperative) {
    LocalStorage::LocalityPlan::Dim d;
    d.indvar = symbolic::symbol("i");
    d.is_gpu = false;
    d.cooperative = cooperative;
    return d;
}

} // namespace

// Sequential For nest: no parallel dims, not inside a kernel.
TEST(LocalStorageTest, Plan_Sequential_NoParallelDims) {
    builder::StructuredSDFGBuilder builder("plan_seq", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto K = symbolic::symbol("K");
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto& loop_i =
        builder.add_for(seq, i, symbolic::Lt(i, K), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)));
    auto& loop_k =
        builder
            .add_for(loop_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    LocalStorage::TileInfo ti;
    ti.bases = {k};
    auto plan = LocalStorage::build_locality_plan(loop_k, ti, am);

    EXPECT_TRUE(plan.dims.empty());
    EXPECT_FALSE(plan.inside_gpu_kernel());
    EXPECT_FALSE(plan.has_gpu_cooperative());
    EXPECT_FALSE(plan.loop_is_outermost);
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Private);
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ true), LocalStorage::Locality::Private);
}

// GPU map whose indvar appears in the tile base → per-thread dim.
TEST(LocalStorageTest, Plan_GpuPerThread) {
    builder::StructuredSDFGBuilder builder("plan_gpu_perthread", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_i =
        builder
            .add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched_x);
    auto& loop_k =
        builder
            .add_for(map_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    LocalStorage::TileInfo ti;
    ti.bases = {symbolic::add(symbolic::mul(i, K), k)}; // base depends on i → per-thread
    auto plan = LocalStorage::build_locality_plan(loop_k, ti, am);

    ASSERT_EQ(plan.dims.size(), 1u);
    EXPECT_TRUE(plan.dims[0].is_gpu);
    EXPECT_FALSE(plan.dims[0].cooperative);
    EXPECT_TRUE(plan.inside_gpu_kernel());
    EXPECT_FALSE(plan.has_gpu_cooperative());
    // Per-thread read or write both localize to a private buffer.
    EXPECT_EQ(LocalStorage::derive_storage(plan, false), LocalStorage::Locality::Private);
    EXPECT_EQ(LocalStorage::derive_storage(plan, true), LocalStorage::Locality::Private);
}

// GPU map whose indvar is absent from the tile base → cooperative dim.
TEST(LocalStorageTest, Plan_GpuCooperative) {
    builder::StructuredSDFGBuilder builder("plan_gpu_coop", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_i =
        builder
            .add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched_x);
    auto& loop_k =
        builder
            .add_for(map_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    LocalStorage::TileInfo ti;
    ti.bases = {k}; // base independent of i → all threads share the tile
    auto plan = LocalStorage::build_locality_plan(loop_k, ti, am);

    ASSERT_EQ(plan.dims.size(), 1u);
    EXPECT_TRUE(plan.dims[0].is_gpu);
    EXPECT_TRUE(plan.dims[0].cooperative);
    EXPECT_TRUE(plan.inside_gpu_kernel());
    EXPECT_TRUE(plan.has_gpu_cooperative());
    EXPECT_FALSE(plan.loop_is_outermost);
    // Cooperative read → shared; cooperative write → reduction we can't lower.
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Shared);
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ true), LocalStorage::Locality::Reject);
}

// =====================================================================
// derive_storage: storage derivation from a schedule classification
// =====================================================================

TEST(LocalStorageTest, Derive_Empty_Private) {
    LocalStorage::LocalityPlan plan;
    EXPECT_EQ(LocalStorage::derive_storage(plan, false), LocalStorage::Locality::Private);
    EXPECT_EQ(LocalStorage::derive_storage(plan, true), LocalStorage::Locality::Private);
}

TEST(LocalStorageTest, Derive_CpuCooperative_Reject) {
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_cpu_dim(/*cooperative*/ true));
    EXPECT_EQ(LocalStorage::derive_storage(plan, false), LocalStorage::Locality::Reject);
    EXPECT_EQ(LocalStorage::derive_storage(plan, true), LocalStorage::Locality::Reject);
}

TEST(LocalStorageTest, Derive_CpuPerThread_Private) {
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_cpu_dim(/*cooperative*/ false));
    EXPECT_EQ(LocalStorage::derive_storage(plan, false), LocalStorage::Locality::Private);
    EXPECT_EQ(LocalStorage::derive_storage(plan, true), LocalStorage::Locality::Private);
}

TEST(LocalStorageTest, Derive_GpuCooperativeRead_Shared) {
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_gpu_dim(/*cooperative*/ true));
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Shared);
}

TEST(LocalStorageTest, Derive_GpuCooperativeWrite_Reject) {
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_gpu_dim(/*cooperative*/ true));
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ true), LocalStorage::Locality::Reject);
}

TEST(LocalStorageTest, Derive_GpuCooperativeOutermost_Reject) {
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_gpu_dim(/*cooperative*/ true));
    plan.loop_is_outermost = true; // a shared buffer can't straddle the kernel boundary
    EXPECT_EQ(LocalStorage::derive_storage(plan, false), LocalStorage::Locality::Reject);
}

TEST(LocalStorageTest, Derive_GpuPerThread_Private) {
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_gpu_dim(/*cooperative*/ false));
    EXPECT_EQ(LocalStorage::derive_storage(plan, false), LocalStorage::Locality::Private);
    EXPECT_EQ(LocalStorage::derive_storage(plan, true), LocalStorage::Locality::Private);
}

// A host-level loop that itself is GPU-scheduled or wraps a GPU kernel is not a
// localization site for a private stack buffer.
TEST(LocalStorageTest, Derive_HostWrapsGpuKernel_Reject) {
    LocalStorage::LocalityPlan plan;
    plan.has_gpu_descendant = true;
    EXPECT_EQ(LocalStorage::derive_storage(plan, false), LocalStorage::Locality::Reject);

    LocalStorage::LocalityPlan plan2;
    plan2.loop_is_gpu = true;
    EXPECT_EQ(LocalStorage::derive_storage(plan2, false), LocalStorage::Locality::Reject);
}

// Cooperation across blocks (grid level) needs grid-wide global memory.
TEST(LocalStorageTest, Derive_GpuGridCooperativeRead_Global) {
    using Level = LocalStorage::LocalityPlan::Level;
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_gpu_dim(/*cooperative*/ true, Level::Grid));
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Global);
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ true), LocalStorage::Locality::Reject);
}

// A read tile with block cooperation lives in shared memory even when it is also
// grid-cooperative: each block redundantly stages its own copy (grid cooperation
// is replication, not a shared buffer). This is the 2D-block GEMM shape.
TEST(LocalStorageTest, Derive_GpuGridAndBlockCooperativeRead_Shared) {
    using Level = LocalStorage::LocalityPlan::Level;
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_gpu_dim(/*cooperative*/ true, Level::Block));
    plan.dims.push_back(make_gpu_dim(/*cooperative*/ true, Level::Grid));
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Shared);
}

// Warp-only cooperation is served by shuffles, not a staged buffer → Reject.
TEST(LocalStorageTest, Derive_GpuWarpCooperativeRead_Reject) {
    using Level = LocalStorage::LocalityPlan::Level;
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_gpu_dim(/*cooperative*/ true, Level::Warp));
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Reject);
}

// A per-thread warp dim (indvar in the tile base) is fine → private register tile.
TEST(LocalStorageTest, Derive_GpuWarpPerThread_Private) {
    using Level = LocalStorage::LocalityPlan::Level;
    LocalStorage::LocalityPlan plan;
    plan.dims.push_back(make_gpu_dim(/*cooperative*/ false, Level::Warp));
    EXPECT_EQ(LocalStorage::derive_storage(plan, false), LocalStorage::Locality::Private);
}

// build_locality_plan reads the new *_Offload schedule: block-level target,
// parallel size, and sync flag.
TEST(LocalStorageTest, Plan_GpuOffload_BlockLevel) {
    builder::StructuredSDFGBuilder builder("plan_offload_block", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(64));
    auto& map_i =
        builder.add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched);
    auto& loop_k =
        builder
            .add_for(map_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    LocalStorage::TileInfo ti;
    ti.bases = {k}; // base independent of i → cooperative across the block
    auto plan = LocalStorage::build_locality_plan(loop_k, ti, am);

    ASSERT_EQ(plan.dims.size(), 1u);
    EXPECT_TRUE(plan.dims[0].is_gpu);
    EXPECT_TRUE(plan.dims[0].cooperative);
    EXPECT_EQ(plan.dims[0].level, LocalStorage::LocalityPlan::Level::Block);
    EXPECT_TRUE(symbolic::eq(plan.dims[0].parallel_size, symbolic::integer(64)));
    EXPECT_FALSE(plan.dims[0].needs_sync);
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Shared);
}

// A grid-level *_Offload cooperative read derives to grid-wide global memory.
TEST(LocalStorageTest, Plan_GpuOffload_GridLevel) {
    builder::StructuredSDFGBuilder builder("plan_offload_grid", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(128));
    auto& map_i =
        builder.add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched);
    auto& loop_k =
        builder
            .add_for(map_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    LocalStorage::TileInfo ti;
    ti.bases = {k};
    auto plan = LocalStorage::build_locality_plan(loop_k, ti, am);

    ASSERT_EQ(plan.dims.size(), 1u);
    EXPECT_EQ(plan.dims[0].level, LocalStorage::LocalityPlan::Level::Grid);
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Global);
}

// A GPU-scheduled Reduce enclosing the loop is a cooperative block level too, so
// build_locality_plan classifies a read tile inside a block reduction as shared
// (previously it saw only Maps and mis-derived a private per-thread buffer).
TEST(LocalStorageTest, Plan_GpuOffloadReduce_BlockLevel) {
    builder::StructuredSDFGBuilder builder("plan_offload_reduce_block", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);

    auto sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(64));
    auto& reduce_i = builder.add_reduce(
        seq,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        sched
    );
    auto& loop_k =
        builder
            .add_for(reduce_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    LocalStorage::TileInfo ti;
    ti.bases = {k}; // base independent of i → cooperative across the block reduce
    auto plan = LocalStorage::build_locality_plan(loop_k, ti, am);

    ASSERT_EQ(plan.dims.size(), 1u);
    EXPECT_TRUE(plan.dims[0].is_gpu);
    EXPECT_TRUE(plan.dims[0].cooperative);
    EXPECT_EQ(plan.dims[0].level, LocalStorage::LocalityPlan::Level::Block);
    EXPECT_TRUE(symbolic::eq(plan.dims[0].parallel_size, symbolic::integer(64)));
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Shared);
}

// A reduction accumulator is owned by the Reduce node + reduce dispatcher;
// A read-only row consumed by Reduce nodes (softmax-style: reduce-max via a
// cmath fmax, reduce-sum, normalize) stages at the enclosing grid loop. The cmath
// input must be no_capture, else the pointer analysis falsely flags X as aliased.
TEST(LocalStorageTest, Gate_ReduceConsumerStaging_Accepts) {
    builder::StructuredSDFGBuilder builder("ls_probe_reduce_consumer", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", elem);
    auto row = symbolic::symbol("row");
    auto jr = symbolic::symbol("jr");
    auto jn = symbolic::symbol("jn");
    auto R = symbolic::symbol("R");
    auto Nc = symbolic::integer(32);
    builder.add_container("R", loop_var, true);
    builder.add_container("X", ptr, true);
    builder.add_container("Y", ptr, true);
    builder.add_container("m", ptr, true);
    builder.add_container("row", loop_var);
    builder.add_container("jr", loop_var);
    builder.add_container("jn", loop_var);

    auto grid = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(1));
    auto block = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));

    auto& map_row =
        builder
            .add_map(seq, row, symbolic::Lt(row, R), symbolic::integer(0), symbolic::add(row, symbolic::integer(1)), grid);
    // reduce-max over the row into m[row], reading X[row*32 + jr]
    auto& reduce = builder.add_reduce(
        map_row.root(),
        jr,
        symbolic::Lt(jr, Nc),
        symbolic::integer(0),
        symbolic::add(jr, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Max, "m"}},
        block
    );
    auto& rblk = builder.add_block(reduce.root());
    auto& m_in = builder.add_access(rblk, "m");
    auto& x_r = builder.add_access(rblk, "X");
    auto& m_out = builder.add_access(rblk, "m");
    auto& rt = builder.add_library_node<
        math::cmath::CMathNode>(rblk, DebugInfo(), math::cmath::CMathFunction::fmax, types::PrimitiveType::Float);
    builder.add_computational_memlet(rblk, m_in, rt, "_in1", {row}, ptr);
    builder.add_computational_memlet(rblk, x_r, rt, "_in2", {symbolic::add(symbolic::mul(row, Nc), jr)}, ptr);
    builder.add_computational_memlet(rblk, rt, "_out", m_out, {row}, ptr);
    // second reduce: sum over the row into s[row], reading X[row*32 + js]
    auto js = symbolic::symbol("js");
    builder.add_container("js", loop_var);
    builder.add_container("s", ptr);
    auto& reduce2 = builder.add_reduce(
        map_row.root(),
        js,
        symbolic::Lt(js, Nc),
        symbolic::integer(0),
        symbolic::add(js, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "s"}},
        block
    );
    auto& r2blk = builder.add_block(reduce2.root());
    auto& s_in = builder.add_access(r2blk, "s");
    auto& x_s = builder.add_access(r2blk, "X");
    auto& s_out = builder.add_access(r2blk, "s");
    auto& st = builder.add_tasklet(r2blk, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(r2blk, s_in, st, "_in1", {row}, ptr);
    builder.add_computational_memlet(r2blk, x_s, st, "_in2", {symbolic::add(symbolic::mul(row, Nc), js)}, ptr);
    builder.add_computational_memlet(r2blk, st, "_out", s_out, {row}, ptr);
    // normalize map reading X[row*32 + jn]
    auto& map_n = builder.add_map(
        map_row.root(), jn, symbolic::Lt(jn, Nc), symbolic::integer(0), symbolic::add(jn, symbolic::integer(1)), block
    );
    auto& nblk = builder.add_block(map_n.root());
    auto& x_n = builder.add_access(nblk, "X");
    auto& y_n = builder.add_access(nblk, "Y");
    builder.add_container("tmp", ptr);
    auto& tmp_o = builder.add_access(nblk, "tmp");
    auto& sub = builder.add_tasklet(nblk, data_flow::TaskletCode::fp_sub, "_out", {"_in1", "_in2"});
    auto& m_n = builder.add_access(nblk, "m");
    builder.add_computational_memlet(nblk, x_n, sub, "_in1", {symbolic::add(symbolic::mul(row, Nc), jn)}, ptr);
    builder.add_computational_memlet(nblk, m_n, sub, "_in2", {row}, ptr);
    builder.add_computational_memlet(nblk, sub, "_out", tmp_o, {symbolic::integer(0)}, ptr);
    auto& tmp_i = builder.add_access(nblk, "tmp");
    auto& s_n = builder.add_access(nblk, "s");
    auto& divt = builder.add_tasklet(nblk, data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(nblk, tmp_i, divt, "_in1", {symbolic::integer(0)}, ptr);
    builder.add_computational_memlet(nblk, s_n, divt, "_in2", {row}, ptr);
    builder.add_computational_memlet(nblk, divt, "_out", y_n, {symbolic::add(symbolic::mul(row, Nc), jn)}, ptr);

    analysis::AnalysisManager am(builder.subject());
    auto plan = LocalStorage::build_locality_plan(map_row, LocalStorage::TileInfo{}, am);
    EXPECT_TRUE(plan.enclosing_cooperative);
    EXPECT_FALSE(LocalStorage::is_reduction_accumulator(map_row, "X", am));
    // A cmath (fmax) reading X must not flag X as pointer-captured (no_capture).
    LocalStorage xform(map_row, x_r);
    EXPECT_TRUE(xform.can_be_applied(builder, am));
}

// is_reduction_accumulator detects it whether the Reduce is the loop itself, an
// ancestor, or a descendant, so can_be_applied refuses to localize it.
TEST(LocalStorageTest, IsReductionAccumulator_EnclosingAndNested) {
    builder::StructuredSDFGBuilder builder("ls_reduce_acc", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);
    builder.add_container("other", ptr);

    auto& for_j =
        builder.add_for(seq, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));
    auto& reduce_i = builder.add_reduce(
        for_j.root(),
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& loop_k =
        builder
            .add_for(reduce_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    EXPECT_TRUE(LocalStorage::is_reduction_accumulator(reduce_i, "acc", am)); // the reduce itself
    EXPECT_TRUE(LocalStorage::is_reduction_accumulator(loop_k, "acc", am)); // ancestor reduce
    EXPECT_TRUE(LocalStorage::is_reduction_accumulator(for_j, "acc", am)); // descendant reduce
    EXPECT_FALSE(LocalStorage::is_reduction_accumulator(loop_k, "other", am));
}

// collect_reduction_owners: a sequential (non-cooperative) Reduce at the localized
// loop is privatizable — it is returned so apply() can retarget its descriptor.
TEST(LocalStorageTest, CollectReductionOwners_SequentialAccepts) {
    builder::StructuredSDFGBuilder builder("ls_collect_seq", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("acc", ptr);

    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_TRUE(LocalStorage::collect_reduction_owners(reduce_j, "acc", am, owners));
    ASSERT_EQ(owners.size(), 1u);
    EXPECT_EQ(owners.front(), &reduce_j);
}

// collect_reduction_owners: a GPU-offloaded (cooperatively combined) Reduce is
// owned by the reduce dispatcher — reject.
TEST(LocalStorageTest, CollectReductionOwners_CooperativeRejects) {
    builder::StructuredSDFGBuilder builder("ls_collect_coop", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("acc", ptr);

    auto block = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        block
    );

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_FALSE(LocalStorage::collect_reduction_owners(reduce_j, "acc", am, owners));
}

// collect_reduction_owners: an *ancestor* Reduce accumulates across iterations
// outside the localized scope — a buffer created at loop_ cannot span it — reject.
TEST(LocalStorageTest, CollectReductionOwners_AncestorRejects) {
    builder::StructuredSDFGBuilder builder("ls_collect_ancestor", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);

    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& loop_k =
        builder
            .add_for(reduce_j.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_FALSE(LocalStorage::collect_reduction_owners(loop_k, "acc", am, owners));
}

// Accumulator privatization: a sequential Reduce over j accumulates y[iO*CY+iI]
// (per-iO block). Localizing y at the Reduce loads the block once, accumulates in
// a Private buffer, writes back once, and retargets the Reduce's descriptor to it
// (gemv --target sequential: LocalStorage(j, y) then vectorize the inner loop).
TEST(LocalStorageTest, Apply_ReductionAccumulator_Sequential) {
    builder::StructuredSDFGBuilder builder("ls_reduce_priv", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", elem);
    auto iO = symbolic::symbol("iO");
    auto j = symbolic::symbol("j");
    auto iI = symbolic::symbol("iI");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    auto CY = symbolic::integer(2);
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("x", ptr, true);
    builder.add_container("y", ptr, true);
    builder.add_container("iO", loop_var);
    builder.add_container("j", loop_var);
    builder.add_container("iI", loop_var);

    auto& for_iO =
        builder.add_for(seq, iO, symbolic::Lt(iO, N), symbolic::integer(0), symbolic::add(iO, symbolic::integer(1)));
    auto& reduce_j = builder.add_reduce(
        for_iO.root(),
        j,
        symbolic::Lt(j, K),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "y"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& for_iI = builder.add_for(
        reduce_j.root(), iI, symbolic::Lt(iI, CY), symbolic::integer(0), symbolic::add(iI, symbolic::integer(1))
    );
    // y[iO*CY + iI] += x[j]
    auto idx = symbolic::add(symbolic::mul(iO, CY), iI);
    auto& blk = builder.add_block(for_iI.root());
    auto& y_in = builder.add_access(blk, "y");
    auto& x_in = builder.add_access(blk, "x");
    auto& y_out = builder.add_access(blk, "y");
    auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(blk, y_in, t, "_in1", {idx}, ptr);
    builder.add_computational_memlet(blk, x_in, t, "_in2", {j}, ptr);
    builder.add_computational_memlet(blk, t, "_out", y_out, {idx}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(reduce_j, y_out); // localize the accumulator at the reduce loop
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    EXPECT_TRUE(xform.storage_type().is_cpu_stack()); // per-thread / sequential private buffer
    xform.apply(builder, am);

    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));

    // The Reduce descriptor now points at the local buffer, matching the rewritten
    // dataflow (denormalized container kept consistent).
    ASSERT_EQ(reduce_j.reductions().size(), 1u);
    EXPECT_EQ(reduce_j.reductions().front().container, buf);

    // The accumulation body reads/writes the buffer, not y; y appears only in the
    // copy-in / writeback maps that now bracket the reduce.
    auto* body = dyn_cast<structured_control_flow::Block*>(&for_iI.root().at(0));
    ASSERT_NE(body, nullptr);
    EXPECT_TRUE(block_uses(*body, buf));
    EXPECT_FALSE(block_uses(*body, "y"));
}

// A cooperatively-combined (GPU-offloaded) reduction accumulator is left to the
// reduce dispatcher — can_be_applied must refuse to localize it.
TEST(LocalStorageTest, CanApply_CooperativeReductionAccumulator_Rejects) {
    builder::StructuredSDFGBuilder builder("ls_reduce_coop_reject", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto row = symbolic::symbol("row");
    auto j = symbolic::symbol("j");
    auto R = symbolic::symbol("R");
    auto Nc = symbolic::integer(32);
    builder.add_container("R", loop_var, true);
    builder.add_container("x", ptr, true);
    builder.add_container("acc", ptr, true);
    builder.add_container("row", loop_var);
    builder.add_container("j", loop_var);

    auto grid = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(1));
    auto block = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    auto& map_row =
        builder
            .add_map(seq, row, symbolic::Lt(row, R), symbolic::integer(0), symbolic::add(row, symbolic::integer(1)), grid);
    // Cooperative block reduction acc[row] += x[row*32 + j].
    auto& reduce_j = builder.add_reduce(
        map_row.root(),
        j,
        symbolic::Lt(j, Nc),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        block
    );
    auto& blk = builder.add_block(reduce_j.root());
    auto& acc_in = builder.add_access(blk, "acc");
    auto& x_in = builder.add_access(blk, "x");
    auto& acc_out = builder.add_access(blk, "acc");
    auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(blk, acc_in, t, "_in1", {row}, ptr);
    builder.add_computational_memlet(blk, x_in, t, "_in2", {symbolic::add(symbolic::mul(row, Nc), j)}, ptr);
    builder.add_computational_memlet(blk, t, "_out", acc_out, {row}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(reduce_j, acc_out);
    EXPECT_FALSE(xform.can_be_applied(builder, am));
}

// =====================================================================
// can_be_applied: schedule gate (end-to-end)
// =====================================================================

// B1: fused-softmax staging topology — a read-only row X[row,:] staged at an
// enclosing grid loop and reused by sibling block loops over the columns.
// build_locality_plan flags this as enclosing_cooperative (a block consumer lives
// below the localized GPU map), deriving a per-block NV_Shared row.
TEST(LocalStorageTest, Gate_EnclosingCooperativeStaging_Accepts) {
    builder::StructuredSDFGBuilder builder("ls_probe_softmax", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", elem);
    auto row = symbolic::symbol("row");
    auto j1 = symbolic::symbol("j1");
    auto j2 = symbolic::symbol("j2");
    auto R = symbolic::symbol("R");
    auto Nc = symbolic::integer(32); // row width: constant so the shared tile is fixed-size
    builder.add_container("R", loop_var, true);
    builder.add_container("X", ptr, true);
    builder.add_container("Y1", ptr, true);
    builder.add_container("Y2", ptr, true);
    builder.add_container("row", loop_var);
    builder.add_container("j1", loop_var);
    builder.add_container("j2", loop_var);

    auto grid = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(1));
    auto block = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));

    auto& map_row =
        builder
            .add_map(seq, row, symbolic::Lt(row, R), symbolic::integer(0), symbolic::add(row, symbolic::integer(1)), grid);
    // Sibling 1: Y1[row*32 + j1] = X[row*32 + j1]
    auto& map_j1 = builder.add_map(
        map_row.root(), j1, symbolic::Lt(j1, Nc), symbolic::integer(0), symbolic::add(j1, symbolic::integer(1)), block
    );
    auto& blk1 = builder.add_block(map_j1.root());
    auto& x_in1 = builder.add_access(blk1, "X");
    auto& y1_out = builder.add_access(blk1, "Y1");
    auto& t1 = builder.add_tasklet(blk1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(blk1, x_in1, t1, "_in", {symbolic::add(symbolic::mul(row, Nc), j1)}, ptr);
    builder.add_computational_memlet(blk1, t1, "_out", y1_out, {symbolic::add(symbolic::mul(row, Nc), j1)}, ptr);
    // Sibling 2: Y2[row*32 + j2] = X[row*32 + j2]
    auto& map_j2 = builder.add_map(
        map_row.root(), j2, symbolic::Lt(j2, Nc), symbolic::integer(0), symbolic::add(j2, symbolic::integer(1)), block
    );
    auto& blk2 = builder.add_block(map_j2.root());
    auto& x_in2 = builder.add_access(blk2, "X");
    auto& y2_out = builder.add_access(blk2, "Y2");
    auto& t2 = builder.add_tasklet(blk2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(blk2, x_in2, t2, "_in", {symbolic::add(symbolic::mul(row, Nc), j2)}, ptr);
    builder.add_computational_memlet(blk2, t2, "_out", y2_out, {symbolic::add(symbolic::mul(row, Nc), j2)}, ptr);

    analysis::AnalysisManager am(builder.subject());

    LocalStorage::TileInfo ti;
    ti.bases = {symbolic::mul(row, Nc)};
    auto plan = LocalStorage::build_locality_plan(map_row, ti, am);
    EXPECT_TRUE(plan.dims.empty());
    EXPECT_TRUE(plan.loop_is_gpu);
    EXPECT_TRUE(plan.has_gpu_descendant);
    EXPECT_TRUE(plan.enclosing_cooperative);
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ false), LocalStorage::Locality::Shared);
    EXPECT_EQ(LocalStorage::derive_storage(plan, /*written*/ true), LocalStorage::Locality::Reject);

    LocalStorage xform(map_row, x_in1);
    EXPECT_TRUE(xform.can_be_applied(builder, am));
}

// Register tiling (thread coarsening): a written accumulator C[iO*CY+iI] under a
// coarse block dim iO, localized at the sequential reduction loop k, becomes a
// per-thread CY-element Private (register) tile — the C_reg role of a GEMM
// micro-kernel. iO is in the tile base (per-thread), so it is not cooperative.
TEST(LocalStorageTest, Apply_RegisterTile_CoarsenedAccumulator) {
    builder::StructuredSDFGBuilder builder("ls_reg_tile", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", elem);
    auto iO = symbolic::symbol("iO");
    auto k = symbolic::symbol("k");
    auto iI = symbolic::symbol("iI");
    auto N = symbolic::symbol("N");
    auto CY = symbolic::integer(2);
    auto K = symbolic::integer(4);
    builder.add_container("N", loop_var, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);
    builder.add_container("iO", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("iI", loop_var);

    auto blocksched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(8));
    auto& map_iO = builder.add_map(
        seq, iO, symbolic::Lt(iO, N), symbolic::integer(0), symbolic::add(iO, symbolic::integer(1)), blocksched
    );
    auto& loop_k =
        builder
            .add_for(map_iO.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));
    auto& loop_iI = builder.add_for(
        loop_k.root(), iI, symbolic::Lt(iI, CY), symbolic::integer(0), symbolic::add(iI, symbolic::integer(1))
    );
    // C[iO*CY + iI] += A[k]
    auto idx = symbolic::add(symbolic::mul(iO, CY), iI);
    auto& blk = builder.add_block(loop_iI.root());
    auto& c_in = builder.add_access(blk, "C");
    auto& a_in = builder.add_access(blk, "A");
    auto& c_out = builder.add_access(blk, "C");
    auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(blk, c_in, t, "_in1", {idx}, ptr);
    builder.add_computational_memlet(blk, a_in, t, "_in2", {k}, ptr);
    builder.add_computational_memlet(blk, t, "_out", c_out, {idx}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop_k, c_out); // localize C at the reduction loop
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    // Per-thread private register tile (not shared).
    EXPECT_FALSE(xform.storage_type().is_nv_shared());
    xform.apply(builder, am);

    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));
    EXPECT_FALSE(builder.subject().type(buf).storage_type().is_nv_shared());
    // The accumulation body (inside the reduction loop) now reads/writes the
    // register tile, not C; C only appears in the copy-in / writeback around it.
    auto* inner_for = dyn_cast<structured_control_flow::For*>(&loop_k.root().at(0));
    ASSERT_NE(inner_for, nullptr);
    auto* body = dyn_cast<structured_control_flow::Block*>(&inner_for->root().at(0));
    ASSERT_NE(body, nullptr);
    EXPECT_TRUE(block_uses(*body, buf));
    EXPECT_FALSE(block_uses(*body, "C"));
}

// Register tiling operand cache: a read-only operand A[iO*CY+iI] that is invariant
// across an inner reuse loop jI is hoisted into a per-thread scalar register,
// loaded once and reused across jI — the a_reg / b_reg role of a GEMM micro-kernel.
TEST(LocalStorageTest, Apply_RegisterTile_OperandReuse) {
    builder::StructuredSDFGBuilder builder("ls_operand_reuse", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", elem);
    auto iO = symbolic::symbol("iO");
    auto iI = symbolic::symbol("iI");
    auto jI = symbolic::symbol("jI");
    auto N = symbolic::symbol("N");
    auto CY = symbolic::integer(2);
    auto CX = symbolic::integer(3);
    builder.add_container("N", loop_var, true);
    builder.add_container("A", ptr, true);
    builder.add_container("Y", ptr, true);
    builder.add_container("iO", loop_var);
    builder.add_container("iI", loop_var);
    builder.add_container("jI", loop_var);

    auto blocksched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(8));
    auto& map_iO = builder.add_map(
        seq, iO, symbolic::Lt(iO, N), symbolic::integer(0), symbolic::add(iO, symbolic::integer(1)), blocksched
    );
    auto& loop_iI = builder.add_for(
        map_iO.root(), iI, symbolic::Lt(iI, CY), symbolic::integer(0), symbolic::add(iI, symbolic::integer(1))
    );
    auto& loop_jI = builder.add_for(
        loop_iI.root(), jI, symbolic::Lt(jI, CX), symbolic::integer(0), symbolic::add(jI, symbolic::integer(1))
    );
    // Y[(iO*CY+iI)*CX + jI] = A[iO*CY+iI]  — A invariant across jI
    auto a_idx = symbolic::add(symbolic::mul(iO, CY), iI);
    auto y_idx = symbolic::add(symbolic::mul(symbolic::add(symbolic::mul(iO, CY), iI), CX), jI);
    auto& blk = builder.add_block(loop_jI.root());
    auto& a_in = builder.add_access(blk, "A");
    auto& y_out = builder.add_access(blk, "Y");
    auto& t = builder.add_tasklet(blk, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(blk, a_in, t, "_in", {a_idx}, ptr);
    builder.add_computational_memlet(blk, t, "_out", y_out, {y_idx}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop_jI, a_in); // cache A at the reuse loop
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    EXPECT_FALSE(xform.storage_type().is_nv_shared()); // private register, not shared

    xform.apply(builder, am);
    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));

    // The load is hoisted above jI: iI body is [copy-in(A->buf), jI-loop], and the
    // jI body reads the register, not A.
    ASSERT_EQ(loop_iI.root().size(), 2u);
    auto* copy_block = copy_block_of(loop_iI.root().at(0));
    ASSERT_NE(copy_block, nullptr);
    EXPECT_TRUE(block_uses(*copy_block, "A"));
    EXPECT_TRUE(block_uses(*copy_block, buf));
    auto* jI_loop = dyn_cast<structured_control_flow::For*>(&loop_iI.root().at(1));
    ASSERT_NE(jI_loop, nullptr);
    auto* body = dyn_cast<structured_control_flow::Block*>(&jI_loop->root().at(0));
    ASSERT_NE(body, nullptr);
    EXPECT_TRUE(block_uses(*body, buf));
    EXPECT_FALSE(block_uses(*body, "A"));
}

// B1 apply: the staged row is loaded once at the top of the grid body (copy map +
// barrier), and BOTH sibling block loops read the shared buffer instead of X.
TEST(LocalStorageTest, Apply_EnclosingCooperativeStaging) {
    builder::StructuredSDFGBuilder builder("ls_apply_softmax", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", elem);
    auto row = symbolic::symbol("row");
    auto j1 = symbolic::symbol("j1");
    auto j2 = symbolic::symbol("j2");
    auto R = symbolic::symbol("R");
    auto Nc = symbolic::integer(32);
    builder.add_container("R", loop_var, true);
    builder.add_container("X", ptr, true);
    builder.add_container("Y1", ptr, true);
    builder.add_container("Y2", ptr, true);
    builder.add_container("row", loop_var);
    builder.add_container("j1", loop_var);
    builder.add_container("j2", loop_var);

    auto grid = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(1));
    auto block = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));

    auto& map_row =
        builder
            .add_map(seq, row, symbolic::Lt(row, R), symbolic::integer(0), symbolic::add(row, symbolic::integer(1)), grid);
    auto& map_j1 = builder.add_map(
        map_row.root(), j1, symbolic::Lt(j1, Nc), symbolic::integer(0), symbolic::add(j1, symbolic::integer(1)), block
    );
    auto& blk1 = builder.add_block(map_j1.root());
    auto& x_in1 = builder.add_access(blk1, "X");
    auto& y1_out = builder.add_access(blk1, "Y1");
    auto& t1 = builder.add_tasklet(blk1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(blk1, x_in1, t1, "_in", {symbolic::add(symbolic::mul(row, Nc), j1)}, ptr);
    builder.add_computational_memlet(blk1, t1, "_out", y1_out, {symbolic::add(symbolic::mul(row, Nc), j1)}, ptr);
    auto& map_j2 = builder.add_map(
        map_row.root(), j2, symbolic::Lt(j2, Nc), symbolic::integer(0), symbolic::add(j2, symbolic::integer(1)), block
    );
    auto& blk2 = builder.add_block(map_j2.root());
    auto& x_in2 = builder.add_access(blk2, "X");
    auto& y2_out = builder.add_access(blk2, "Y2");
    auto& t2 = builder.add_tasklet(blk2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(blk2, x_in2, t2, "_in", {symbolic::add(symbolic::mul(row, Nc), j2)}, ptr);
    builder.add_computational_memlet(blk2, t2, "_out", y2_out, {symbolic::add(symbolic::mul(row, Nc), j2)}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(map_row, x_in1);
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    EXPECT_TRUE(xform.storage_type().is_nv_shared());
    xform.apply(builder, am);

    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));
    EXPECT_TRUE(builder.subject().type(buf).storage_type().is_nv_shared());

    // Grid body is now [copy_map (offload), barrier, map_j1, map_j2].
    ASSERT_EQ(map_row.root().size(), 4u);
    auto* copy_map = dyn_cast<structured_control_flow::Map*>(&map_row.root().at(0));
    ASSERT_NE(copy_map, nullptr);
    EXPECT_EQ(copy_map->schedule_type().category(), structured_control_flow::ScheduleTypeCategory::Offloader);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&map_row.root().at(1)), nullptr); // barrier
    EXPECT_NE(dyn_cast<structured_control_flow::Map*>(&map_row.root().at(2)), nullptr);
    EXPECT_NE(dyn_cast<structured_control_flow::Map*>(&map_row.root().at(3)), nullptr);

    // Both consumers now read the shared buffer, not X.
    EXPECT_TRUE(block_uses(blk1, buf));
    EXPECT_FALSE(block_uses(blk1, "X"));
    EXPECT_TRUE(block_uses(blk2, buf));
    EXPECT_FALSE(block_uses(blk2, "X"));
}

/**
 * Gate_GpuCooperativeRead_Rejects: a read-only tile that is cooperative across a
 * GPU thread dim derives to shared memory, whose apply path is not implemented
 * yet — so can_be_applied rejects (rather than mis-lowering to a private stack).
 */
TEST(LocalStorageTest, Gate_GpuCooperativeRead_Rejects) {
    builder::StructuredSDFGBuilder builder("ls_gate_gpu_coop", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched, symbolic::integer(32));
    auto& map_i =
        builder.add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched);
    auto& loop_k = builder.add_for(
        map_i.root(),
        k,
        symbolic::Lt(k, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );

    // C[i] = A[k] — A is read cooperatively (base independent of thread dim i).
    auto& block = builder.add_block(loop_k.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, t, "_in", {k}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {i}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop_k, a_in);
    EXPECT_FALSE(xform.can_be_applied(builder, am));
}

/**
 * Gate_GpuPerThread_Accepts: a per-thread tile (base uses the GPU thread dim)
 * derives to a private stack buffer and is accepted — register blocking inside a
 * kernel.
 */
TEST(LocalStorageTest, Gate_GpuPerThread_Accepts) {
    builder::StructuredSDFGBuilder builder("ls_gate_gpu_perthread", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched, symbolic::integer(32));
    auto& map_i =
        builder.add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched);
    auto& loop_k = builder.add_for(
        map_i.root(),
        k,
        symbolic::Lt(k, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );

    // C[i] = A[i*16 + k] — A tile is per-thread (base uses thread dim i).
    auto& block = builder.add_block(loop_k.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, a_in, t, "_in", {symbolic::add(symbolic::mul(i, symbolic::integer(16)), k)}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {i}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop_k, a_in);
    EXPECT_TRUE(xform.can_be_applied(builder, am));
    EXPECT_TRUE(xform.storage_type().is_cpu_stack());
}

/**
 * Apply_Cooperative_Shared: a read-only tile cooperative across a GPU block dim
 * (new *_Offload schedule) localizes into an NV_Shared buffer, staged by a
 * flattened offload copy-Map + a barrier, with the body reading the buffer.
 */
TEST(LocalStorageTest, Apply_Cooperative_Shared) {
    builder::StructuredSDFGBuilder builder("ls_coop_shared", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    auto& map_i =
        builder.add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched);
    auto& loop_k = builder.add_for(
        map_i.root(),
        k,
        symbolic::Lt(k, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );

    // out[i] += A[k] — A[k] is read cooperatively (base independent of thread dim i).
    auto& block = builder.add_block(loop_k.root());
    auto& c_in = builder.add_access(block, "C");
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {i}, ptr);
    builder.add_computational_memlet(block, a_in, t, "_in2", {k}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {i}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop_k, a_in);
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    EXPECT_TRUE(xform.storage_type().is_nv_shared());
    xform.apply(builder, am);

    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));
    EXPECT_TRUE(builder.subject().type(buf).storage_type().is_nv_shared());

    // The kernel-map body is now [copy_map (offload), barrier, k-loop].
    ASSERT_EQ(map_i.root().size(), 3u);
    auto* copy_map = dyn_cast<structured_control_flow::Map*>(&map_i.root().at(0));
    ASSERT_NE(copy_map, nullptr);
    EXPECT_EQ(copy_map->schedule_type().category(), structured_control_flow::ScheduleTypeCategory::Offloader);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&map_i.root().at(1)), nullptr); // barrier block
    EXPECT_NE(dyn_cast<structured_control_flow::For*>(&map_i.root().at(2)), nullptr);

    // The body reads the shared buffer, not A.
    auto* main_block = dyn_cast<structured_control_flow::Block*>(&loop_k.root().at(0));
    ASSERT_NE(main_block, nullptr);
    EXPECT_TRUE(block_uses(*main_block, buf));
    EXPECT_FALSE(block_uses(*main_block, "A"));
}

// =====================================================================
// Pure buffer / tile index math (TileBuffer, TileInfo) — no SDFG state
// =====================================================================

TEST(LocalStorageTest, TileBuffer_TotalSize) {
    LocalStorage::TileBuffer buf{{symbolic::integer(2)}, {symbolic::integer(3), symbolic::integer(4)}};
    EXPECT_TRUE(symbolic::eq(buf.total_size(), symbolic::integer(24)));

    LocalStorage::TileBuffer tile_only{{}, {symbolic::integer(8)}};
    EXPECT_TRUE(symbolic::eq(tile_only.total_size(), symbolic::integer(8)));
}

TEST(LocalStorageTest, TileBuffer_Linearize_TileOnly) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    LocalStorage::TileBuffer buf{{}, {symbolic::integer(3), symbolic::integer(4)}};
    // Row-major: i*4 + j.
    EXPECT_TRUE(symbolic::eq(buf.linearize({}, {i, j}), symbolic::add(symbolic::mul(i, symbolic::integer(4)), j)));
}

TEST(LocalStorageTest, TileBuffer_Linearize_WithSlots) {
    auto s = symbolic::symbol("s");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    LocalStorage::TileBuffer buf{{symbolic::integer(2)}, {symbolic::integer(3), symbolic::integer(4)}};
    // Row-major over [slot(2), tile(3,4)]: s*12 + i*4 + j.
    auto expected =
        symbolic::add(symbolic::mul(s, symbolic::integer(12)), symbolic::add(symbolic::mul(i, symbolic::integer(4)), j));
    EXPECT_TRUE(symbolic::eq(buf.linearize({s}, {i, j}), expected));
}

TEST(LocalStorageTest, TileBuffer_Delinearize) {
    auto c = symbolic::symbol("c");
    // Single tile dim: the flat index passes through unchanged.
    LocalStorage::TileBuffer d1{{}, {symbolic::integer(4)}};
    auto one = d1.delinearize_tile(c);
    ASSERT_EQ(one.size(), 1u);
    EXPECT_TRUE(symbolic::eq(one[0], c));

    // Two tile dims (3 x 4): row-major decomposition [c / 4, c % 4] (integer
    // div/mod, lowered by codegen). Tested structurally — symbolic::div is exact
    // division, so a concrete roundtrip would not floor.
    LocalStorage::TileBuffer d2{{}, {symbolic::integer(3), symbolic::integer(4)}};
    auto two = d2.delinearize_tile(c);
    ASSERT_EQ(two.size(), 2u);
    EXPECT_TRUE(symbolic::eq(two[0], symbolic::div(c, symbolic::integer(4))));
    EXPECT_TRUE(symbolic::eq(two[1], symbolic::mod(c, symbolic::integer(4))));
}

TEST(LocalStorageTest, TileInfo_VaryingDims) {
    LocalStorage::TileInfo ti;
    ti.dimensions = {symbolic::integer(1), symbolic::integer(8), symbolic::integer(1), symbolic::integer(4)};
    EXPECT_EQ(ti.varying_dims(), (std::vector<size_t>{1, 3}));
    auto sizes = ti.varying_sizes();
    ASSERT_EQ(sizes.size(), 2u);
    EXPECT_TRUE(symbolic::eq(sizes[0], symbolic::integer(8)));
    EXPECT_TRUE(symbolic::eq(sizes[1], symbolic::integer(4)));
}

TEST(LocalStorageTest, TileInfo_OriginalSubset_StridedBox) {
    // 2D box: dim0 degenerate (row i), dim1 varying (extent 4), strides [M, 1], offset off.
    auto i = symbolic::symbol("i");
    auto M = symbolic::symbol("M");
    auto off = symbolic::symbol("off");
    auto j = symbolic::symbol("j");
    LocalStorage::TileInfo ti;
    ti.dimensions = {symbolic::integer(1), symbolic::integer(4)};
    ti.bases = {i, symbolic::integer(0)};
    ti.strides = {M, symbolic::integer(1)};
    ti.offset = off;
    // original_subset({j}) = off + M*i + (0 + j).
    auto res = ti.original_subset({j});
    ASSERT_EQ(res.size(), 1u);
    EXPECT_TRUE(symbolic::eq(res[0], symbolic::add(off, symbolic::add(symbolic::mul(M, i), j))));
}

TEST(LocalStorageTest, TileInfo_LocalIndex) {
    // Only the varying dim contributes a local index (access - base).
    auto i = symbolic::symbol("i");
    auto c = symbolic::symbol("c");
    auto x = symbolic::symbol("x");
    LocalStorage::TileInfo ti;
    ti.dimensions = {symbolic::integer(1), symbolic::integer(4)};
    ti.bases = {i, c};
    auto local = ti.local_index({i, x});
    ASSERT_EQ(local.size(), 1u);
    EXPECT_TRUE(symbolic::eq(local[0], symbolic::sub(x, c)));
}

TEST(LocalStorageTest, TileBuffer_SlotOffset) {
    // [slot(2)][tile(3,4)]: tile block = 12, so slot s starts at s*12.
    LocalStorage::TileBuffer buf{{symbolic::integer(2)}, {symbolic::integer(3), symbolic::integer(4)}};
    EXPECT_TRUE(symbolic::eq(buf.tile_total_size(), symbolic::integer(12)));
    auto s = symbolic::symbol("s");
    EXPECT_TRUE(symbolic::eq(buf.slot_offset({s}), symbolic::mul(s, symbolic::integer(12))));

    // [slot(2,5)][tile(4)]: row-major slots scaled by tile_total 4 -> a*20 + b*4.
    LocalStorage::TileBuffer buf2{{symbolic::integer(2), symbolic::integer(5)}, {symbolic::integer(4)}};
    EXPECT_TRUE(symbolic::eq(buf2.tile_total_size(), symbolic::integer(4)));
    auto a = symbolic::symbol("a");
    auto b = symbolic::symbol("b");
    EXPECT_TRUE(symbolic::
                    eq(buf2.slot_offset({a, b}),
                       symbolic::add(symbolic::mul(a, symbolic::integer(20)), symbolic::mul(b, symbolic::integer(4)))));

    // slot_offset(slots) + tile-linear == full linearize(slots, tile) (consistency).
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    EXPECT_TRUE(symbolic::
                    eq(buf.linearize({s}, {i, j}),
                       symbolic::add(buf.slot_offset({s}), symbolic::add(symbolic::mul(i, symbolic::integer(4)), j))));
}

/**
 * Apply_Cooperative_Mixed: a tile that is per-thread along one GPU block dim (i)
 * and cooperative along another (j) — shared-memory GEMM shape. The buffer gains
 * a per-thread slot prefix (sized by i's block width), the copy is staged with a
 * leading + trailing barrier, and the body reads buf[threadIdx.x-slot][k].
 */
TEST(LocalStorageTest, Apply_Cooperative_Mixed) {
    builder::StructuredSDFGBuilder builder("ls_coop_mixed", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    builder.add_container("N", loop_var, true);
    builder.add_container("M", loop_var, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);
    builder.add_container("i", loop_var);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);

    auto sched_i = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(8));
    auto sched_j = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::Y_BLOCK, symbolic::integer(4));
    auto& map_i =
        builder
            .add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched_i);
    auto& map_j = builder.add_map(
        map_i.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)), sched_j
    );
    auto& loop_k = builder.add_for(
        map_j.root(),
        k,
        symbolic::Lt(k, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );

    // C[i*M + j] += A[i*16 + k] — A per-thread in i (base uses i), cooperative in j.
    auto& block = builder.add_block(loop_k.root());
    auto& c_in = builder.add_access(block, "C");
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {symbolic::add(symbolic::mul(i, M), j)}, ptr);
    builder
        .add_computational_memlet(block, a_in, t, "_in2", {symbolic::add(symbolic::mul(i, symbolic::integer(16)), k)}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {symbolic::add(symbolic::mul(i, M), j)}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop_k, a_in);
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    EXPECT_TRUE(xform.storage_type().is_nv_shared());
    xform.apply(builder, am);

    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));
    // Buffer = [slot(BM=8)] x [tile(16)] = 128 shared elements.
    EXPECT_TRUE(builder.subject().type(buf).storage_type().is_nv_shared());
    EXPECT_TRUE(builder.subject().type(buf) == types::Array(elem, symbolic::integer(128)));

    // The cooperative map body is [leading barrier, copy_map, trailing barrier, k-loop].
    ASSERT_EQ(map_j.root().size(), 4u);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&map_j.root().at(0)), nullptr);
    auto* copy_map = dyn_cast<structured_control_flow::Map*>(&map_j.root().at(1));
    ASSERT_NE(copy_map, nullptr);
    EXPECT_EQ(copy_map->schedule_type().category(), structured_control_flow::ScheduleTypeCategory::Offloader);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&map_j.root().at(2)), nullptr);
    EXPECT_NE(dyn_cast<structured_control_flow::For*>(&map_j.root().at(3)), nullptr);

    // The body reads the shared buffer, not A.
    auto* main_block = dyn_cast<structured_control_flow::Block*>(&loop_k.root().at(0));
    ASSERT_NE(main_block, nullptr);
    EXPECT_TRUE(block_uses(*main_block, buf));
    EXPECT_FALSE(block_uses(*main_block, "A"));
}

/**
 * Apply_Cooperative_Mixed_CoopOuter: the 2D-block GEMM shape. The tile is
 * cooperative along the OUTER GPU block dim (j) and per-thread along the INNER,
 * immediately-enclosing dim (i). This is exactly what LocalStorage v1 rejected
 * (it required the cooperative axis to be the immediate parent). v2 must accept
 * it AND parallelize the copy over the cooperative axis (j, width 4) — not the
 * per-thread immediate parent (i, width 8); using i would stride the copy along
 * the slot axis and leave each slot only partially filled.
 */
TEST(LocalStorageTest, Apply_Cooperative_Mixed_CoopOuter) {
    builder::StructuredSDFGBuilder builder("ls_coop_mixed_outer", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    builder.add_container("N", loop_var, true);
    builder.add_container("M", loop_var, true);
    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);
    builder.add_container("i", loop_var);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);

    // j is the OUTER cooperative axis (X_BLOCK, width 4); i is the INNER,
    // immediately-enclosing per-thread axis (Y_BLOCK, width 8).
    auto sched_j = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(4));
    auto sched_i = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::Y_BLOCK, symbolic::integer(8));
    auto& map_j =
        builder
            .add_map(seq, j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)), sched_j);
    auto& map_i = builder.add_map(
        map_j.root(), i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched_i
    );
    auto& loop_k = builder.add_for(
        map_i.root(),
        k,
        symbolic::Lt(k, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );

    // C[i*M + j] += A[i*16 + k] — A per-thread in i (base uses i), cooperative in j.
    auto& block = builder.add_block(loop_k.root());
    auto& c_in = builder.add_access(block, "C");
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& t = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, t, "_in1", {symbolic::add(symbolic::mul(i, M), j)}, ptr);
    builder
        .add_computational_memlet(block, a_in, t, "_in2", {symbolic::add(symbolic::mul(i, symbolic::integer(16)), k)}, ptr);
    builder.add_computational_memlet(block, t, "_out", c_out, {symbolic::add(symbolic::mul(i, M), j)}, ptr);

    analysis::AnalysisManager am(builder.subject());
    LocalStorage xform(loop_k, a_in);
    ASSERT_TRUE(xform.can_be_applied(builder, am));
    EXPECT_TRUE(xform.storage_type().is_nv_shared());
    xform.apply(builder, am);

    auto buf = xform.local_container();
    ASSERT_TRUE(builder.subject().exists(buf));
    // Buffer = [slot(i width = 8)] x [tile(16)] = 128 shared elements.
    EXPECT_TRUE(builder.subject().type(buf).storage_type().is_nv_shared());
    EXPECT_TRUE(builder.subject().type(buf) == types::Array(elem, symbolic::integer(128)));

    // The copy sits in the immediately-enclosing (per-thread) map's body:
    // [leading barrier, copy_map, trailing barrier, k-loop].
    ASSERT_EQ(map_i.root().size(), 4u);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&map_i.root().at(0)), nullptr);
    auto* copy_map = dyn_cast<structured_control_flow::Map*>(&map_i.root().at(1));
    ASSERT_NE(copy_map, nullptr);
    EXPECT_EQ(copy_map->schedule_type().category(), structured_control_flow::ScheduleTypeCategory::Offloader);
    // The copy must be parallelized over the cooperative axis j (X_BLOCK, width 4),
    // NOT the per-thread immediate parent i (Y_BLOCK, width 8).
    EXPECT_EQ(gpu::gpu_target_level(copy_map->schedule_type()), gpu::TargetLevel::X_BLOCK);
    EXPECT_TRUE(symbolic::eq(gpu::ScheduleType_GPU_Offload::parallel_size(copy_map->schedule_type()), symbolic::integer(4))
    );
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&map_i.root().at(2)), nullptr);
    EXPECT_NE(dyn_cast<structured_control_flow::For*>(&map_i.root().at(3)), nullptr);

    // The body reads the shared buffer, not A.
    auto* main_block_outer = dyn_cast<structured_control_flow::Block*>(&loop_k.root().at(0));
    ASSERT_NE(main_block_outer, nullptr);
    EXPECT_TRUE(block_uses(*main_block_outer, buf));
    EXPECT_FALSE(block_uses(*main_block_outer, "A"));
}
