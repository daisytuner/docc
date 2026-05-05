#include "sdfg/transformations/in_local_storage.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/transformations/utils.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"

using namespace sdfg;

/**
 * Test: InLocalStorage on a scalar array (1D) with constant bound
 *
 * Before:
 *   for i = 0..4: C += A[i]
 *
 * After:
 *   for i' = 0..4: A_local[i'] = A[i']
 *   for i = 0..4: C += A_local[i]
 */
TEST(InLocalStorage, Scalar_ConstantBound) {
    builder::StructuredSDFGBuilder builder("ils_scalar_test", FunctionType_CPU);

    // Create containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true); // read-only input
    builder.add_container("C", elem_desc); // accumulator

    auto& root = builder.subject().root();

    // Create loop: for i = 0..4
    auto indvar = symbolic::symbol("i");
    auto bound = symbolic::integer(4);
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation: C += A[i]
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply transformation
    transformations::InLocalStorage transformation(loop, a_in);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    // Verify: local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));

    // Verify: structure should now be [copy_loop, main_loop]
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 2);

    // First element should be copy loop
    auto* copy_loop = dynamic_cast<structured_control_flow::Map*>(&new_root.at(0).first);
    EXPECT_NE(copy_loop, nullptr);

    // Second element should be the main loop
    auto* main_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(main_loop, nullptr);

    // Verify copy loop reads from A, writes to A_local
    auto& copy_body = copy_loop->root();
    EXPECT_EQ(copy_body.size(), 1);
    auto* copy_block = dynamic_cast<structured_control_flow::Block*>(&copy_body.at(0).first);
    EXPECT_NE(copy_block, nullptr);

    bool reads_A = false;
    bool writes_A_local = false;
    for (auto* node : copy_block->dataflow().data_nodes()) {
        if (node->data() == "A") reads_A = true;
        if (node->data() == "__daisy_in_local_storage_A0") writes_A_local = true;
    }
    EXPECT_TRUE(reads_A);
    EXPECT_TRUE(writes_A_local);

    // Verify main loop uses local buffer
    auto& main_body = main_loop->root();
    EXPECT_EQ(main_body.size(), 1);
    auto* main_block = dynamic_cast<structured_control_flow::Block*>(&main_body.at(0).first);
    EXPECT_NE(main_block, nullptr);

    bool uses_A_local = false;
    bool uses_A_original = false;
    for (auto* node : main_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_in_local_storage_A0") uses_A_local = true;
        if (node->data() == "A") uses_A_original = true;
    }
    EXPECT_TRUE(uses_A_local);
    EXPECT_FALSE(uses_A_original); // Original A should be replaced
}

/**
 * Test: InLocalStorage should fail on containers that are written
 *
 * for i = 0..N: C[i] += A[i]
 *
 * InLocalStorage(loop, "C") should fail because C is written
 * InLocalStorage(loop, "A") should succeed because A is read-only
 */
TEST(InLocalStorage, FailsOnWrittenContainer) {
    builder::StructuredSDFGBuilder builder("ils_rw_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true); // read-only
    builder.add_container("C", ptr_desc, true); // read-write

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // C[i] += A[i]
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {indvar}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {indvar}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on C since it's written
    transformations::InLocalStorage ils_c(loop, c_in);
    EXPECT_FALSE(ils_c.can_be_applied(builder_opt, am));

    // InLocalStorage should SUCCEED on A since A is read-only
    transformations::InLocalStorage ils_a(loop, a_in);
    EXPECT_TRUE(ils_a.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage should fail on scalars
 *
 * for i = 0..N: ... scalar_val ...
 *
 * InLocalStorage(loop, "scalar_val") should fail because it's not an array
 */
TEST(InLocalStorage, FailsOnScalar) {
    builder::StructuredSDFGBuilder builder("ils_scalar_fail_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    builder.add_container("scalar_val", elem_desc, true);
    builder.add_container("result", elem_desc);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // result = result + scalar_val
    auto& block = builder.add_block(loop.root());
    auto& s_in = builder.add_access(block, "scalar_val");
    auto& r_in = builder.add_access(block, "result");
    auto& r_out = builder.add_access(block, "result");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, r_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, s_in, tasklet, "_in2", {}, elem_desc);
    builder.add_computational_memlet(block, tasklet, "_out", r_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on scalar_val
    transformations::InLocalStorage ils(loop, s_in);
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage should fail when access node is outside the loop
 */
TEST(InLocalStorage, FailsOnAccessOutsideLoop) {
    builder::StructuredSDFGBuilder builder("ils_outside_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc, true);

    auto& root = builder.subject().root();

    // Place an access to B outside the loop
    auto& outer_block = builder.add_block(root);
    auto& b_outside = builder.add_access(outer_block, "B");
    auto& i_outside = builder.add_access(outer_block, "i");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(outer_block, i_outside, tasklet_outside, "_in", {});
    builder.add_computational_memlet(outer_block, tasklet_outside, "_out", b_outside, {symbolic::integer(0)}, ptr_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on B (not used inside the loop)
    transformations::InLocalStorage ils(loop, b_outside);
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage should fail when container is not used
 */
TEST(InLocalStorage, FailsOnUnusedContainer) {
    builder::StructuredSDFGBuilder builder("ils_unused_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc, true); // declared but not used

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // Place an access to B outside the loop
    auto& outer_block = builder.add_block(root);
    auto& b_outside = builder.add_access(outer_block, "B");
    auto& i_outside = builder.add_access(outer_block, "i");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(outer_block, i_outside, tasklet_outside, "_in", {});
    builder.add_computational_memlet(outer_block, tasklet_outside, "_out", b_outside, {symbolic::integer(0)}, ptr_desc);

    // Only use A, not B inside the loop
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on B (not used in loop body)
    transformations::InLocalStorage ils(loop, b_outside);
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: JSON serialization round-trip
 */
TEST(InLocalStorage, JsonSerialization) {
    builder::StructuredSDFGBuilder builder("ils_json_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true);
    builder.add_container("C", elem_desc);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    analysis::AnalysisManager am(builder.subject());

    // Create transformation and serialize
    transformations::InLocalStorage original(loop, a_in);
    EXPECT_TRUE(original.can_be_applied(builder, am));

    nlohmann::json j;
    original.to_json(j);

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "InLocalStorage");
    EXPECT_EQ(j["container"], "A");
    EXPECT_TRUE(j.contains("subgraph"));

    // Deserialize and verify
    auto deserialized = transformations::InLocalStorage::from_json(builder, j);
    EXPECT_EQ(deserialized.name(), "InLocalStorage");
    EXPECT_TRUE(deserialized.can_be_applied(builder, am));
}

/**
 * Test: InLocalStorage on 2D tiled access (post-tiling of a symbolic loop nest)
 *
 * Before (after tiling i by MC=64, j by NC=32):
 *   for i_tile = 0..M step MC:
 *       for j_tile = 0..N step NC:
 *           for i = i_tile..min(i_tile+MC, M):
 *               for j = j_tile..min(j_tile+NC, N):
 *                   C += A[i*N+j]
 *
 * After InLocalStorage(i_loop, "A"):
 *   Tile at i_loop level has extents MC x NC (integer after overapprox).
 *   for i_tile = 0..M step MC:
 *       for j_tile = 0..N step NC:
 *           for d0 = 0..MC: for d1 = 0..NC:
 *               A_local[d0*NC+d1] = A[(i_tile+d0)*N+(j_tile+d1)]
 *           for i = i_tile..min(i_tile+MC, M):
 *               for j = j_tile..min(j_tile+NC, N):
 *                   C += A_local[(i-i_tile)*NC+(j-j_tile)]
 *
 * Buffer size: MC * NC (constant, known at compile time)
 */
TEST(InLocalStorage, TiledAccess_2D) {
    builder::StructuredSDFGBuilder builder("ils_2d_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("j_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("C", elem_desc);

    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true);

    auto& root = builder.subject().root();

    auto MC = symbolic::integer(64);
    auto NC = symbolic::integer(32);
    auto M = symbolic::symbol("M");
    auto N = symbolic::symbol("N");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i_tile = 0; i_tile < M; i_tile += MC
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, M), symbolic::integer(0), symbolic::add(i_tile, MC));

    // for j_tile = 0; j_tile < N; j_tile += NC
    auto& j_tile_loop =
        builder
            .add_for(i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, N), symbolic::integer(0), symbolic::add(j_tile, NC));

    // for i = i_tile; i < min(i_tile+MC, M); i++
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, MC)), symbolic::Lt(i, M)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for j = j_tile; j < min(j_tile+NC, N); j++
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, NC)), symbolic::Lt(j, N)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    // C += A[i*N+j]
    auto& block = builder.add_block(j_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::add(symbolic::mul(i, N), j)}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at i_loop level (tile has extents MC x NC)
    transformations::InLocalStorage ils(i_loop, a_in);

    bool can_apply = ils.can_be_applied(builder_opt, am);
    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));

        // Verify: j_tile_loop body should be [copy_loop, i_loop]
        auto& j_tile_body = j_tile_loop.root();
        EXPECT_EQ(j_tile_body.size(), 2u);

        // First: copy loop (outer dim, 0..MC)
        auto* copy_loop = dynamic_cast<structured_control_flow::Map*>(&j_tile_body.at(0).first);
        EXPECT_NE(copy_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(copy_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(copy_loop->condition(), symbolic::Lt(copy_loop->indvar(), MC)));

        // Second: compute loop (i_loop preserved)
        auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&j_tile_body.at(1).first);
        EXPECT_NE(compute_loop, nullptr);

        // Verify the compute memlet uses LOCAL indices: (i-i_tile)*NC + (j-j_tile)
        auto& compute_i_body = compute_loop->root();
        EXPECT_GE(compute_i_body.size(), 1u);
        auto* compute_j_loop = dynamic_cast<structured_control_flow::For*>(&compute_i_body.at(0).first);
        EXPECT_NE(compute_j_loop, nullptr);
        auto& compute_j_body = compute_j_loop->root();
        EXPECT_EQ(compute_j_body.size(), 1u);
        auto* compute_block = dynamic_cast<structured_control_flow::Block*>(&compute_j_body.at(0).first);
        EXPECT_NE(compute_block, nullptr);

        bool found_local_access = false;
        for (auto* node : compute_block->dataflow().data_nodes()) {
            if (node->data() == "__daisy_in_local_storage_A0") {
                found_local_access = true;
                for (auto& memlet : compute_block->dataflow().out_edges(*node)) {
                    auto& subset = memlet.subset();
                    EXPECT_EQ(subset.size(), 1u);
                    // Expected: (i - i_tile) * NC + (j - j_tile)
                    auto expected =
                        symbolic::add(symbolic::mul(symbolic::sub(i, i_tile), NC), symbolic::sub(j, j_tile));
                    EXPECT_TRUE(symbolic::eq(subset.at(0), expected));
                }
            }
        }
        EXPECT_TRUE(found_local_access);
    }
}

/**
 * Test: InLocalStorage with 1D tiled access (post-tiling scenario)
 *
 * Before (after tiling i by TILE=64):
 *   for i_tile = 0..N step TILE:
 *       for i = i_tile..min(i_tile+TILE, N):
 *           C += A[i]
 *
 * After InLocalStorage(inner_loop, "A"):
 *   Tile at inner_loop level has extent TILE (integer after overapprox).
 *   for i_tile = 0..N step TILE:
 *       for d0 = 0..TILE:
 *           A_local[d0] = A[i_tile + d0]
 *       for i = i_tile..min(i_tile+TILE, N):
 *           C += A_local[i - i_tile]
 */
TEST(InLocalStorage, TiledAccess_1D) {
    builder::StructuredSDFGBuilder builder("ils_tiled_1d_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("C", elem_desc);

    // A is a 1D pointer: double* A (size N)
    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true);

    auto& root = builder.subject().root();

    // Tile size
    auto TILE = symbolic::integer(64);
    auto N = symbolic::symbol("N");
    auto i_tile = symbolic::symbol("i_tile");
    auto i = symbolic::symbol("i");

    // Outer loop: for i_tile = 0; i_tile < N; i_tile += TILE
    auto& tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, TILE));

    // Inner loop: for i = i_tile; i < min(i_tile+TILE, N); i++
    auto& inner_loop = builder.add_for(
        tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, TILE)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // C += A[i]
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {i}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at inner_loop level (tile has extent TILE)
    transformations::InLocalStorage ils(inner_loop, a_in);

    bool can_apply = ils.can_be_applied(builder_opt, am);

    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));

        // Verify: tile_loop body should be [copy_loop, inner_loop]
        EXPECT_EQ(tile_loop.root().size(), 2u);

        // First: copy loop (0..TILE)
        auto* copy_loop = dynamic_cast<structured_control_flow::Map*>(&tile_loop.root().at(0).first);
        EXPECT_NE(copy_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(copy_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(copy_loop->condition(), symbolic::Lt(copy_loop->indvar(), TILE)));
    }
}

/**
 * Test: InLocalStorage with 2D tiled access (BLIS-style panel packing)
 *
 * Before (after tiling i by MC=64, k by KC=64):
 *   for i_tile = 0..M step MC:
 *       for k_tile = 0..K step KC:
 *           for i = i_tile..min(i_tile+MC, M):
 *               for k = k_tile..min(k_tile+KC, K):
 *                   ... = A[i*K+k]
 *
 * After InLocalStorage(i_loop, "A"):
 *   Tile at i_loop level has extents MC x KC (integer after overapprox).
 *   for i_tile = 0..M step MC:
 *       for k_tile = 0..K step KC:
 *           for d0 = 0..MC: for d1 = 0..KC:
 *               A_local[d0*KC+d1] = A[(i_tile+d0)*K+(k_tile+d1)]
 *           for i = i_tile..min(i_tile+MC, M):
 *               for k = k_tile..min(k_tile+KC, K):
 *                   ... = A_local[(i-i_tile)*KC+(k-k_tile)]
 *
 * Buffer size: MC * KC (constant, known at compile time)
 */
TEST(InLocalStorage, TiledAccess_2D_Panel) {
    builder::StructuredSDFGBuilder builder("ils_tiled_2d_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("M", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("k_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("k", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("C", elem_desc);

    // A is a 2D array: double A[M][K] (linearized as A[i*K+k])
    types::Pointer a_desc(elem_desc);
    builder.add_container("A", a_desc, true);

    auto& root = builder.subject().root();

    // Tile sizes
    auto MC = symbolic::integer(64);
    auto KC = symbolic::integer(64);
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    auto i_tile = symbolic::symbol("i_tile");
    auto k_tile = symbolic::symbol("k_tile");
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");

    // Outer: for i_tile = 0; i_tile < M; i_tile += MC
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, M), symbolic::integer(0), symbolic::add(i_tile, MC));

    // Next: for k_tile = 0; k_tile < K; k_tile += KC
    auto& k_tile_loop =
        builder
            .add_for(i_tile_loop.root(), k_tile, symbolic::Lt(k_tile, K), symbolic::integer(0), symbolic::add(k_tile, KC));

    // Inner: for i = i_tile; i < i_tile + MC; i++
    auto& i_loop = builder.add_for(
        k_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, MC)), symbolic::Lt(i, M)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // Innermost: for k = k_tile; k < k_tile + KC; k++
    auto& k_loop = builder.add_for(
        i_loop.root(),
        k,
        symbolic::And(symbolic::Lt(k, symbolic::add(k_tile, KC)), symbolic::Lt(k, K)),
        k_tile,
        symbolic::add(k, symbolic::one())
    );

    // C += A[i][k]
    auto& block = builder.add_block(k_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::add(symbolic::mul(i, K), k)}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at i_loop level (tile has extents MC x KC)
    transformations::InLocalStorage ils(i_loop, a_in);

    bool can_apply = ils.can_be_applied(builder_opt, am);

    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));

        // Verify: k_tile_loop body should be [copy_loop, i_loop]
        EXPECT_EQ(k_tile_loop.root().size(), 2u);

        // First: copy loop (outer dim, 0..MC)
        auto* copy_loop = dynamic_cast<structured_control_flow::Map*>(&k_tile_loop.root().at(0).first);
        EXPECT_NE(copy_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(copy_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(copy_loop->condition(), symbolic::Lt(copy_loop->indvar(), MC)));

        // Check nested second dimension (0..KC)
        auto& copy_inner_body = copy_loop->root();
        EXPECT_EQ(copy_inner_body.size(), 1u);
        auto* copy_inner = dynamic_cast<structured_control_flow::Map*>(&copy_inner_body.at(0).first);
        EXPECT_NE(copy_inner, nullptr);
        EXPECT_TRUE(symbolic::eq(copy_inner->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(copy_inner->condition(), symbolic::Lt(copy_inner->indvar(), KC)));

        // Verify the compute memlet uses LOCAL indices: (i-i_tile)*KC + (k-k_tile)
        auto& compute_i_body = i_loop.root();
        EXPECT_GE(compute_i_body.size(), 1u);
        auto* compute_k_loop = dynamic_cast<structured_control_flow::For*>(&compute_i_body.at(0).first);
        EXPECT_NE(compute_k_loop, nullptr);
        auto& compute_k_body = compute_k_loop->root();
        EXPECT_EQ(compute_k_body.size(), 1u);
        auto* compute_block = dynamic_cast<structured_control_flow::Block*>(&compute_k_body.at(0).first);
        EXPECT_NE(compute_block, nullptr);

        bool found_local_access = false;
        for (auto* node : compute_block->dataflow().data_nodes()) {
            if (node->data() == "__daisy_in_local_storage_A0") {
                found_local_access = true;
                for (auto& memlet : compute_block->dataflow().out_edges(*node)) {
                    auto& subset = memlet.subset();
                    EXPECT_EQ(subset.size(), 1u);
                    // Expected: (i - i_tile) * KC + (k - k_tile)
                    auto expected =
                        symbolic::add(symbolic::mul(symbolic::sub(i, i_tile), KC), symbolic::sub(k, k_tile));
                    EXPECT_TRUE(symbolic::eq(subset.at(0), expected));
                }
            }
        }
        EXPECT_TRUE(found_local_access);
    }
}

/**
 * Test: InLocalStorage with 2D tiled stencil access (halo region)
 *
 * A 5-point stencil on A[N][M] (linearized as A[i*M+j]) with rectangular tiling.
 *
 * Before (after tiling i by IT=32, j by JT=32):
 *   for i_tile = 1..N-1 step IT:
 *       for j_tile = 1..M-1 step JT:
 *           for i = i_tile..min(i_tile+IT, N-1):
 *               for j = j_tile..min(j_tile+JT, M-1):
 *                   B[i*M+j] = A[(i-1)*M+j] + A[(i+1)*M+j]
 *                            + A[i*M+(j-1)] + A[i*M+(j+1)] + A[i*M+j]
 *
 * After InLocalStorage(i_loop, "A"):
 *   Tile at i_loop level has extents (IT+2) x (JT+2) due to stencil halo.
 *   The stencil accesses i-1..i+IT (overapproximate of i range plus +/-1 halo)
 *   and j-1..j+JT, giving integer extents.
 */
TEST(InLocalStorage, TiledStencil_2D_5Point) {
    builder::StructuredSDFGBuilder builder("ils_tiled_stencil_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("j_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc, true);

    auto& root = builder.subject().root();

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i_tile = symbolic::symbol("i_tile");
    auto j_tile = symbolic::symbol("j_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto IT = symbolic::integer(32);
    auto JT = symbolic::integer(32);

    auto N_minus_1 = symbolic::sub(N, symbolic::one());
    auto M_minus_1 = symbolic::sub(M, symbolic::one());

    // for i_tile = 1; i_tile < N-1; i_tile += IT
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N_minus_1), symbolic::one(), symbolic::add(i_tile, IT));

    // for j_tile = 1; j_tile < M-1; j_tile += JT
    auto& j_tile_loop = builder.add_for(
        i_tile_loop.root(), j_tile, symbolic::Lt(j_tile, M_minus_1), symbolic::one(), symbolic::add(j_tile, JT)
    );

    // for i = i_tile; i < min(i_tile+IT, N-1); i++
    auto& i_loop = builder.add_for(
        j_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, IT)), symbolic::Lt(i, N_minus_1)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for j = j_tile; j < min(j_tile+JT, M-1); j++
    auto& j_loop = builder.add_for(
        i_loop.root(),
        j,
        symbolic::And(symbolic::Lt(j, symbolic::add(j_tile, JT)), symbolic::Lt(j, M_minus_1)),
        j_tile,
        symbolic::add(j, symbolic::one())
    );

    // 5-point stencil: B[i*M+j] = A[(i-1)*M+j] + A[(i+1)*M+j] + A[i*M+(j-1)] + A[i*M+(j+1)] + A[i*M+j]
    // Each stencil point in its own block with fp_add accumulation into B
    auto& block = builder.add_block(j_loop.root());

    auto center = symbolic::add(symbolic::mul(i, M), j);
    auto north = symbolic::add(symbolic::mul(symbolic::sub(i, symbolic::one()), M), j);
    auto south = symbolic::add(symbolic::mul(symbolic::add(i, symbolic::one()), M), j);
    auto west = symbolic::add(symbolic::mul(i, M), symbolic::sub(j, symbolic::one()));
    auto east = symbolic::add(symbolic::mul(i, M), symbolic::add(j, symbolic::one()));

    // B[center] = A[center]
    auto& a_center = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& t0 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_center, t0, "_in", {center});
    builder.add_computational_memlet(block, t0, "_out", b_out, {center});

    // B[center] += A[north]
    auto& block_n = builder.add_block(j_loop.root());
    auto& a_north = builder.add_access(block_n, "A");
    auto& b_in_n = builder.add_access(block_n, "B");
    auto& b_out_n = builder.add_access(block_n, "B");
    auto& t_n = builder.add_tasklet(block_n, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_n, b_in_n, t_n, "_in1", {center});
    builder.add_computational_memlet(block_n, a_north, t_n, "_in2", {north});
    builder.add_computational_memlet(block_n, t_n, "_out", b_out_n, {center});

    // B[center] += A[south]
    auto& block_s = builder.add_block(j_loop.root());
    auto& a_south = builder.add_access(block_s, "A");
    auto& b_in_s = builder.add_access(block_s, "B");
    auto& b_out_s = builder.add_access(block_s, "B");
    auto& t_s = builder.add_tasklet(block_s, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_s, b_in_s, t_s, "_in1", {center});
    builder.add_computational_memlet(block_s, a_south, t_s, "_in2", {south});
    builder.add_computational_memlet(block_s, t_s, "_out", b_out_s, {center});

    // B[center] += A[west]
    auto& block_w = builder.add_block(j_loop.root());
    auto& a_west = builder.add_access(block_w, "A");
    auto& b_in_w = builder.add_access(block_w, "B");
    auto& b_out_w = builder.add_access(block_w, "B");
    auto& t_w = builder.add_tasklet(block_w, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_w, b_in_w, t_w, "_in1", {center});
    builder.add_computational_memlet(block_w, a_west, t_w, "_in2", {west});
    builder.add_computational_memlet(block_w, t_w, "_out", b_out_w, {center});

    // B[center] += A[east]
    auto& block_e = builder.add_block(j_loop.root());
    auto& a_east = builder.add_access(block_e, "A");
    auto& b_in_e = builder.add_access(block_e, "B");
    auto& b_out_e = builder.add_access(block_e, "B");
    auto& t_e = builder.add_tasklet(block_e, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block_e, b_in_e, t_e, "_in1", {center});
    builder.add_computational_memlet(block_e, a_east, t_e, "_in2", {east});
    builder.add_computational_memlet(block_e, t_e, "_out", b_out_e, {center});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at i_loop level for A (tile has extents (IT+2) x (JT+2))
    transformations::InLocalStorage ils(i_loop, a_center);

    bool can_apply = ils.can_be_applied(builder_opt, am);

    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));

        // Verify: j_tile_loop body should be [copy_loop, i_loop]
        EXPECT_EQ(j_tile_loop.root().size(), 2u);
    }
}

// =========================================================================
// GPU Cooperative Path Tests
// =========================================================================

/**
 * Test: InLocalStorage with NV_Shared rejects when no cooperative dimension exists
 *
 * Setup: GPU Map X (i, 0..N) → For k = 0..K, accessing A[i*K + k]
 * Tile base depends on i (the only GPU dim), so no cooperative dim → rejected.
 */
TEST(InLocalStorage, GPU_NoCoop_Rejected) {
    builder::StructuredSDFGBuilder builder("ils_gpu_nocoop", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");

    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr);
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    // Single GPU map (X dim): i = 0..N (block_size=32)
    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_x = builder.add_map(
        seq,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), N),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        sched_x
    );

    // For loop k = 0..K
    auto& loop = builder.add_for(
        map_x.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), K),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // A[i*K + k] — linearized, base depends on i (GPU indvar)
    builder.add_computational_memlet(
        block, a_in, tasklet, "_in", {symbolic::add(symbolic::mul(symbolic::symbol("i"), K), symbolic::symbol("k"))}, ptr
    );
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("i")}, ptr);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // NV_Shared with no cooperative dimension → should be rejected
    // (Also K is unresolvable, but coop check comes first after extent resolution)
    transformations::InLocalStorage ils(loop, a_in, types::StorageType::NV_Shared());
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage with NV_Shared on a flat pointer with cooperative loading
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..16
 *   A[i*16 + k] — linearized pointer access
 *
 * N, M are symbolic (resolved via schedule: N=32, M=8).
 * Tile bases = [i*16], extent = [16] (constant — no substitution needed here).
 * Y-dim (j) does NOT appear in base → cooperative.
 */
TEST(InLocalStorage, GPU_Cooperative_FlatPointer) {
    builder::StructuredSDFGBuilder builder("ils_gpu_coop", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr);
    builder.add_container("N", loop_var, true);
    builder.add_container("M", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);

    // GPU Map X: i = 0..N (block_size=32)
    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_x = builder.add_map(
        seq,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), N),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        sched_x
    );

    // GPU Map Y: j = 0..M (block_size=8)
    auto sched_y = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(sched_y, cuda::CUDADimension::Y);
    gpu::gpu_block_size(sched_y, symbolic::integer(8));
    auto& map_y = builder.add_map(
        map_x.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), M),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        sched_y
    );

    // For loop: k = 0..16
    auto& loop = builder.add_for(
        map_y.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // A[i*16 + k] — base depends on i only; j is free → cooperative
    builder.add_computational_memlet(
        block,
        a_in,
        tasklet,
        "_in",
        {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::integer(16)), symbolic::symbol("k"))},
        ptr
    );
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("j")}, ptr);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::InLocalStorage ils(loop, a_in, types::StorageType::NV_Shared());
    EXPECT_TRUE(ils.can_be_applied(builder_opt, am));
    ils.apply(builder_opt, am);

    // Verify: shared buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));
    auto& buf_type = builder_opt.subject().type("__daisy_in_local_storage_A0");
    EXPECT_EQ(buf_type.storage_type(), types::StorageType::NV_Shared());

    // Verify structure: [barrier, copy_loop, barrier, main_loop]
    auto& map_y_body = map_y.root();
    EXPECT_GE(map_y_body.size(), 4u);

    // First element: barrier block
    auto* barrier1 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(0).first);
    EXPECT_NE(barrier1, nullptr);
    bool has_barrier1 = false;
    for (auto& node : barrier1->dataflow().nodes()) {
        if (auto* lib = dynamic_cast<data_flow::LibraryNode*>(&node)) {
            if (lib->code() == data_flow::LibraryNodeType_BarrierLocal) {
                has_barrier1 = true;
            }
        }
    }
    EXPECT_TRUE(has_barrier1);

    // Second element: cooperative copy map (strided loop)
    auto* copy_map = dynamic_cast<structured_control_flow::Map*>(&map_y_body.at(1).first);
    EXPECT_NE(copy_map, nullptr);

    // Third element: barrier block
    auto* barrier2 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(2).first);
    EXPECT_NE(barrier2, nullptr);
    bool has_barrier2 = false;
    for (auto& node : barrier2->dataflow().nodes()) {
        if (auto* lib = dynamic_cast<data_flow::LibraryNode*>(&node)) {
            if (lib->code() == data_flow::LibraryNodeType_BarrierLocal) {
                has_barrier2 = true;
            }
        }
    }
    EXPECT_TRUE(has_barrier2);

    // Fourth element: original for loop
    auto* main_loop = dynamic_cast<structured_control_flow::For*>(&map_y_body.at(3).first);
    EXPECT_NE(main_loop, nullptr);
}

/**
 * Test: InLocalStorage with NV_Shared and symbolic GPU map bounds
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..M
 *   A[i*M + k] — linearized flat pointer
 *
 * N and M are symbolic, but GPU schedule says N=32, M=8.
 * Tile for A over For k: bases = [i*M], extents = [M]. M is symbolic.
 * After substitution: M→8, extent becomes 8 (integer). j-dim free → cooperative.
 */
TEST(InLocalStorage, GPU_Cooperative_SymbolicBounds) {
    builder::StructuredSDFGBuilder builder("ils_gpu_symbolic", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr);
    builder.add_container("N", loop_var, true);
    builder.add_container("M", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);

    // GPU Map X: i = 0..N (block_size=32)
    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_x = builder.add_map(
        seq,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), N),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        sched_x
    );

    // GPU Map Y: j = 0..M (block_size=8)
    auto sched_y = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(sched_y, cuda::CUDADimension::Y);
    gpu::gpu_block_size(sched_y, symbolic::integer(8));
    auto& map_y = builder.add_map(
        map_x.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), M),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        sched_y
    );

    // For loop: k = 0..M (same symbol as Y-dim bound)
    auto& loop = builder.add_for(
        map_y.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), M),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // A[i*M + k] — linearized access with symbolic stride M
    builder.add_computational_memlet(
        block, a_in, tasklet, "_in", {symbolic::add(symbolic::mul(symbolic::symbol("i"), M), symbolic::symbol("k"))}, ptr
    );
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("j")}, ptr);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::InLocalStorage ils(loop, a_in, types::StorageType::NV_Shared());
    EXPECT_TRUE(ils.can_be_applied(builder_opt, am));
    ils.apply(builder_opt, am);

    // Verify: shared buffer with resolved size (M→8)
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));
    auto& buf_type = builder_opt.subject().type("__daisy_in_local_storage_A0");
    EXPECT_EQ(buf_type.type_id(), types::TypeID::Array);
    EXPECT_EQ(buf_type.storage_type(), types::StorageType::NV_Shared());

    auto& arr_type = static_cast<const types::Array&>(buf_type);
    // Extent M resolved to 8 from GPU Y-dim block_size
    EXPECT_TRUE(symbolic::eq(arr_type.num_elements(), symbolic::integer(8)));

    // Verify structure: [barrier, copy_loop, barrier, main_loop]
    auto& map_y_body = map_y.root();
    EXPECT_GE(map_y_body.size(), 4u);

    auto* barrier1 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(0).first);
    EXPECT_NE(barrier1, nullptr);
    auto* copy_map = dynamic_cast<structured_control_flow::Map*>(&map_y_body.at(1).first);
    EXPECT_NE(copy_map, nullptr);
    auto* barrier2 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(2).first);
    EXPECT_NE(barrier2, nullptr);
    auto* main_loop = dynamic_cast<structured_control_flow::For*>(&map_y_body.at(3).first);
    EXPECT_NE(main_loop, nullptr);
}

/**
 * Test: InLocalStorage with NV_Shared rejects when symbolic extent can't be resolved
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..K
 *   A[i*K + k] — K is NOT a GPU dim bound, can't be resolved
 *
 * Extent = K (symbolic, unresolvable) → rejected.
 */
TEST(InLocalStorage, GPU_SymbolicExtent_Unresolvable_Rejected) {
    builder::StructuredSDFGBuilder builder("ils_gpu_unresolved", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");

    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr);
    builder.add_container("N", loop_var, true);
    builder.add_container("M", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);

    // GPU Map X: i = 0..N (block_size=32)
    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_x = builder.add_map(
        seq,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), N),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        sched_x
    );

    // GPU Map Y: j = 0..M (block_size=8)
    auto sched_y = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(sched_y, cuda::CUDADimension::Y);
    gpu::gpu_block_size(sched_y, symbolic::integer(8));
    auto& map_y = builder.add_map(
        map_x.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), M),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        sched_y
    );

    // For loop: k = 0..K (K is NOT a GPU block size — unresolvable)
    auto& loop = builder.add_for(
        map_y.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), K),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // A[i*K + k] — linearized with symbolic stride K, extent K (unresolvable)
    builder.add_computational_memlet(
        block, a_in, tasklet, "_in", {symbolic::add(symbolic::mul(symbolic::symbol("i"), K), symbolic::symbol("k"))}, ptr
    );
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("j")}, ptr);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // K can't be resolved from any GPU block size → rejected
    transformations::InLocalStorage ils(loop, a_in, types::StorageType::NV_Shared());
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage with NV_Shared and both GPU dims cooperative
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..N
 *   A[k] — access does not depend on any GPU dim
 *
 * Tile bases = [0], extents = [N]. N is symbolic but resolvable to 32.
 * Neither i nor j appear in bases → both cooperative.
 */
TEST(InLocalStorage, GPU_Cooperative_AllDimsFree) {
    builder::StructuredSDFGBuilder builder("ils_gpu_allfree", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");

    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr);
    builder.add_container("N", loop_var, true);
    builder.add_container("M", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);

    // GPU Map X: i = 0..N (block_size=32)
    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_x = builder.add_map(
        seq,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), N),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        sched_x
    );

    // GPU Map Y: j = 0..M (block_size=8)
    auto sched_y = cuda::ScheduleType_CUDA::create();
    cuda::ScheduleType_CUDA::dimension(sched_y, cuda::CUDADimension::Y);
    gpu::gpu_block_size(sched_y, symbolic::integer(8));
    auto& map_y = builder.add_map(
        map_x.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), M),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        sched_y
    );

    // For loop: k = 0..N (same bound as X-dim)
    auto& loop = builder.add_for(
        map_y.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), N),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    // A[k] — no GPU indvar in access
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("k")}, ptr);
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        c_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("i"), M), symbolic::symbol("j"))},
        ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::InLocalStorage ils(loop, a_in, types::StorageType::NV_Shared());
    EXPECT_TRUE(ils.can_be_applied(builder_opt, am));
    ils.apply(builder_opt, am);

    // Verify buffer: extent N resolved to 32
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));
    auto& buf_type = builder_opt.subject().type("__daisy_in_local_storage_A0");
    auto& arr_type = static_cast<const types::Array&>(buf_type);
    EXPECT_TRUE(symbolic::eq(arr_type.num_elements(), symbolic::integer(32)));
    EXPECT_EQ(buf_type.storage_type(), types::StorageType::NV_Shared());
}

/**
 * Test: InLocalStorage CPU_Stack with flat pointer (non-GPU baseline)
 *
 * Ensures the transformation still works correctly on CPU with flat pointers.
 *
 * Setup: for k = 0..16: C += A[i*16 + k] (flat pointer access)
 * After: Map(0..16) copies A[i*16+d] → local[d], for loop uses local[k]
 */
TEST(InLocalStorage, CPU_FlatPointer_Linearized) {
    builder::StructuredSDFGBuilder builder("ils_cpu_flatptr", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    builder.add_container("A", ptr, true);
    builder.add_container("C", elem);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");

    // Outer loop: i = 0..100
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(100)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    // Inner loop: k = 0..16
    auto& loop = builder.add_for(
        outer_loop.root(),
        k,
        symbolic::Lt(k, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    // A[i*16 + k] — flat pointer linearized access
    builder.add_computational_memlet(
        block, a_in, tasklet, "_in2", {symbolic::add(symbolic::mul(i, symbolic::integer(16)), k)}, ptr
    );
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::InLocalStorage ils(loop, a_in);
    EXPECT_TRUE(ils.can_be_applied(builder_opt, am));
    ils.apply(builder_opt, am);

    // Verify: buffer created, structure inside outer loop = [copy_map, main_loop]
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));
    auto& outer_body = outer_loop.root();
    EXPECT_EQ(outer_body.size(), 2u);

    auto* copy_map = dynamic_cast<structured_control_flow::Map*>(&outer_body.at(0).first);
    EXPECT_NE(copy_map, nullptr);
    // Copy map iterates 0..16
    EXPECT_TRUE(symbolic::eq(copy_map->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(copy_map->condition(), symbolic::Lt(copy_map->indvar(), symbolic::integer(16))));

    auto* main_loop = dynamic_cast<structured_control_flow::For*>(&outer_body.at(1).first);
    EXPECT_NE(main_loop, nullptr);

    // Verify the compute memlet uses LOCAL indices (k, zero-based)
    auto& main_body = main_loop->root();
    EXPECT_EQ(main_body.size(), 1u);
    auto* compute_block = dynamic_cast<structured_control_flow::Block*>(&main_body.at(0).first);
    EXPECT_NE(compute_block, nullptr);

    bool found_local_access = false;
    for (auto* node : compute_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_in_local_storage_A0") {
            found_local_access = true;
            for (auto& memlet : compute_block->dataflow().out_edges(*node)) {
                // After ILS, the subset should be {k} (local index, zero-based)
                auto& subset = memlet.subset();
                EXPECT_EQ(subset.size(), 1u);
                EXPECT_TRUE(symbolic::eq(subset.at(0), k));
            }
        }
    }
    EXPECT_TRUE(found_local_access);
}

// =========================================================================
// Tile Group Tests: Multiple access groups to the same container
// =========================================================================

/**
 * Test: InLocalStorage with SYR2K-style pattern A[i*K+k] and A[j*K+k]
 *
 * Two independent read accesses to the same container with different row indices.
 * ILS applied to A[i,k] group should:
 *   - Only pack that group (1×K tile)
 *   - Leave A[j,k] access unchanged
 *   - After first ILS, second ILS on A[j,k] can be applied independently
 */
TEST(InLocalStorage, TileGroups_TwoIndependentReads) {
    builder::StructuredSDFGBuilder builder("ils_syr2k_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", elem_desc);

    auto& root = builder.subject().root();

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto K = symbolic::integer(16);

    // for i = 0..N
    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // for j = 0..N
    auto& j_loop =
        builder.add_for(i_loop.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // for k = 0..16
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // Block: tmp = A[i*16 + k] * A[j*16 + k] (fp_mul has 2 inputs)
    auto& block = builder.add_block(k_loop.root());
    auto& a_ik = builder.add_access(block, "A");
    auto& a_jk = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_a", "_b"});

    auto lin_ik = symbolic::add(symbolic::mul(i, K), k);
    auto lin_jk = symbolic::add(symbolic::mul(j, K), k);

    builder.add_computational_memlet(block, a_ik, tasklet, "_a", {lin_ik}, ptr_desc);
    builder.add_computational_memlet(block, a_jk, tasklet, "_b", {lin_jk}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply ILS on the A[i,k] access node at k_loop level
    transformations::InLocalStorage ils_ik(k_loop, a_ik);
    EXPECT_TRUE(ils_ik.can_be_applied(builder_opt, am));
    ils_ik.apply(builder_opt, am);

    // Verify: local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));

    // j_loop body: [k_loop] but k_loop's parent (j_loop.root()) now has [copy_map, k_loop]
    auto& j_body = j_loop.root();
    EXPECT_EQ(j_body.size(), 2u);

    auto* copy_map = dynamic_cast<structured_control_flow::Map*>(&j_body.at(0).first);
    EXPECT_NE(copy_map, nullptr);
    // Copy should iterate 0..16 (the K dimension)
    EXPECT_TRUE(symbolic::eq(copy_map->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(copy_map->condition(), symbolic::Lt(copy_map->indvar(), K)));

    // Verify compute block: one access should be rewritten to local, one should remain original
    auto& k_body = k_loop.root();
    EXPECT_EQ(k_body.size(), 1u);
    auto* compute_block = dynamic_cast<structured_control_flow::Block*>(&k_body.at(0).first);
    ASSERT_NE(compute_block, nullptr);

    bool found_local = false;
    bool found_original = false;
    for (auto* node : compute_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_in_local_storage_A0") {
            found_local = true;
            // Local access should have zero-based index {k}
            for (auto& memlet : compute_block->dataflow().out_edges(*node)) {
                auto& subset = memlet.subset();
                EXPECT_EQ(subset.size(), 1u);
                EXPECT_TRUE(symbolic::eq(subset.at(0), k));
            }
        } else if (node->data() == "A") {
            found_original = true;
            // Original A[j*16+k] access should be unchanged
            for (auto& memlet : compute_block->dataflow().out_edges(*node)) {
                auto& subset = memlet.subset();
                EXPECT_EQ(subset.size(), 1u);
                EXPECT_TRUE(symbolic::eq(subset.at(0), lin_jk));
            }
        }
    }
    EXPECT_TRUE(found_local);
    EXPECT_TRUE(found_original);
}

/**
 * Test: InLocalStorage applied twice for both groups independently
 *
 * After applying ILS on A[i,k], apply ILS on A[j,k] as a second transformation.
 * Both should succeed, creating two separate local buffers.
 */
TEST(InLocalStorage, TileGroups_SequentialApplication) {
    builder::StructuredSDFGBuilder builder("ils_syr2k_seq_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", elem_desc);

    auto& root = builder.subject().root();

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto K = symbolic::integer(16);

    // for i = 0..N
    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // for j = 0..N
    auto& j_loop =
        builder.add_for(i_loop.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // for k = 0..16
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // Block: tmp = A[i*16 + k] * A[j*16 + k] (fp_mul has 2 inputs)
    auto& block = builder.add_block(k_loop.root());
    auto& a_ik = builder.add_access(block, "A");
    auto& a_jk = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_a", "_b"});

    auto lin_ik = symbolic::add(symbolic::mul(i, K), k);
    auto lin_jk = symbolic::add(symbolic::mul(j, K), k);

    builder.add_computational_memlet(block, a_ik, tasklet, "_a", {lin_ik}, ptr_desc);
    builder.add_computational_memlet(block, a_jk, tasklet, "_b", {lin_jk}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // First ILS: pack A[i,k] group
    transformations::InLocalStorage ils_ik(k_loop, a_ik);
    ASSERT_TRUE(ils_ik.can_be_applied(builder_opt, am));
    ils_ik.apply(builder_opt, am);

    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A0"));

    // Second ILS: pack A[j,k] group — the remaining "A" access node
    // Find it in the compute block
    auto& k_body = k_loop.root();
    auto* compute_block = dynamic_cast<structured_control_flow::Block*>(&k_body.at(0).first);
    ASSERT_NE(compute_block, nullptr);

    const data_flow::AccessNode* remaining_a = nullptr;
    for (auto* node : compute_block->dataflow().data_nodes()) {
        if (node->data() == "A") {
            remaining_a = node;
            break;
        }
    }
    ASSERT_NE(remaining_a, nullptr);

    transformations::InLocalStorage ils_jk(k_loop, *remaining_a);
    EXPECT_TRUE(ils_jk.can_be_applied(builder_opt, am));
    ils_jk.apply(builder_opt, am);

    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A1"));

    // After both: no "A" access nodes should remain in the k_loop body
    auto& k_body2 = k_loop.root();
    auto* compute2 = dynamic_cast<structured_control_flow::Block*>(&k_body2.at(0).first);
    ASSERT_NE(compute2, nullptr);

    for (auto* node : compute2->dataflow().data_nodes()) {
        EXPECT_NE(node->data(), "A") << "No raw 'A' accesses should remain after both ILS applications";
    }

    // j_loop body should have: [copy_map_1, copy_map_2, k_loop]
    auto& j_body = j_loop.root();
    EXPECT_EQ(j_body.size(), 3u);
}
