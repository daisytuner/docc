#include "sdfg/transformations/in_local_storage.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/utils.h"

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
    types::Array a_desc(elem_desc, symbolic::integer(4));
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
    transformations::InLocalStorage transformation(loop, "A");
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    // Verify: local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A"));

    // Verify: structure should now be [copy_loop, main_loop]
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 2);

    // First element should be copy loop
    auto* copy_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
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
        if (node->data() == "__daisy_in_local_storage_A") writes_A_local = true;
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
        if (node->data() == "__daisy_in_local_storage_A") uses_A_local = true;
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
    types::Array arr_desc(elem_desc, symbolic::integer(4));
    builder.add_container("A", arr_desc, true); // read-only
    builder.add_container("C", arr_desc, true); // read-write

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
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {indvar}, arr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {indvar}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on C since it's written
    transformations::InLocalStorage ils_c(loop, "C");
    EXPECT_FALSE(ils_c.can_be_applied(builder_opt, am));

    // InLocalStorage should SUCCEED on A since A is read-only
    transformations::InLocalStorage ils_a(loop, "A");
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
    transformations::InLocalStorage ils(loop, "scalar_val");
    EXPECT_FALSE(ils.can_be_applied(builder_opt, am));
}

/**
 * Test: InLocalStorage should fail when container doesn't exist
 */
TEST(InLocalStorage, FailsOnNonexistent) {
    builder::StructuredSDFGBuilder builder("ils_nonexist_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Array arr_desc(elem_desc, symbolic::integer(4));
    builder.add_container("A", arr_desc, true);

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
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on nonexistent container
    transformations::InLocalStorage ils(loop, "B_nonexistent");
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
    types::Array arr_desc(elem_desc, symbolic::integer(4));
    builder.add_container("A", arr_desc, true);
    builder.add_container("B", arr_desc, true); // declared but not used

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // Only use A, not B
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, arr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // InLocalStorage should FAIL on B (not used in loop body)
    transformations::InLocalStorage ils(loop, "B");
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
    types::Array arr_desc(elem_desc, symbolic::integer(4));
    builder.add_container("A", arr_desc, true);
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, arr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    analysis::AnalysisManager am(builder.subject());

    // Create transformation and serialize
    transformations::InLocalStorage original(loop, "A");
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
 * Test: InLocalStorage on 2D array with nested loops (both dimensions vary)
 *
 * Before:
 *   for i = 0..M:
 *       for j = 0..N:
 *           C += A[i][j]
 *
 * After InLocalStorage(outer_loop, "A"):
 *   for i' = 0..M:
 *       for j' = 0..N:
 *           A_local[i'][j'] = A[i'][j']
 *   for i = 0..M:
 *       for j = 0..N:
 *           C += A_local[i][j]
 *
 * Buffer size: M * N (linearized)
 */
TEST(InLocalStorage, NestedLoops_2D) {
    builder::StructuredSDFGBuilder builder("ils_2d_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("C", elem_desc);

    // A is a 2D array: double A[M][N]
    types::Array a_row(elem_desc, symbolic::symbol("N"));
    types::Pointer a_desc(a_row);
    builder.add_container("A", a_desc, true);

    auto& root = builder.subject().root();

    // Outer loop: for i = 0..M
    auto i = symbolic::symbol("i");
    auto M = symbolic::symbol("M");
    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, M), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)));

    // Inner loop: for j = 0..N
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto& j_loop =
        builder
            .add_for(i_loop.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    // C += A[i][j]
    auto& block = builder.add_block(j_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {i, j}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at outer loop level
    transformations::InLocalStorage ils(i_loop, "A");

    std::cout << "Testing InLocalStorage on 2D nested loops..." << std::endl;
    bool can_apply = ils.can_be_applied(builder_opt, am);
    std::cout << "  can_be_applied: " << (can_apply ? "yes" : "no") << std::endl;

    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A"));

        // Verify: should have copy loops + main loop
        auto& new_root = builder_opt.subject().root();
        EXPECT_GE(new_root.size(), 2u);

        std::cout << "  Local buffer created: yes" << std::endl;
        std::cout << "  Root elements after: " << new_root.size() << std::endl;
    }
}

/**
 * Test: InLocalStorage with tiled indices (simulating post-tiling scenario)
 *
 * Before (after tiling):
 *   for i_tile = 0..N step TILE:
 *       for i = i_tile..i_tile+TILE:
 *           C += A[i]
 *
 * After InLocalStorage(i_tile_loop, "A"):
 *   for i_tile = 0..N step TILE:
 *       for i' = 0..TILE:
 *           A_local[i'] = A[i_tile + i']
 *       for i = i_tile..i_tile+TILE:
 *           C += A_local[i - i_tile]
 *
 * This tests the key BLIS packing scenario.
 */
TEST(InLocalStorage, TiledAccess_1D) {
    builder::StructuredSDFGBuilder builder("ils_tiled_1d_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    builder.add_container("C", elem_desc);

    // A is a 1D array: double A[N]
    types::Array a_desc(elem_desc, symbolic::symbol("N"));
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
    // Simplified to: for i = i_tile; i < i_tile+TILE; i++
    auto& inner_loop = builder.add_for(
        tile_loop.root(), i, symbolic::Lt(i, symbolic::add(i_tile, TILE)), i_tile, symbolic::add(i, symbolic::integer(1))
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

    // Apply InLocalStorage at tile loop level
    transformations::InLocalStorage ils(tile_loop, "A");

    std::cout << "Testing InLocalStorage on tiled 1D access..." << std::endl;
    bool can_apply = ils.can_be_applied(builder_opt, am);
    std::cout << "  can_be_applied: " << (can_apply ? "yes" : "no") << std::endl;

    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A"));

        // Verify: tile_loop should now have copy loop + inner_loop
        EXPECT_GE(tile_loop.root().size(), 2u);

        std::cout << "  Local buffer created: yes" << std::endl;
        std::cout << "  Tile loop children after: " << tile_loop.root().size() << std::endl;
    }
}

/**
 * Test: InLocalStorage with 2D tiled access (BLIS-style panel)
 *
 * Before:
 *   for i_tile = 0..M step MC:
 *       for k_tile = 0..K step KC:
 *           for i = i_tile..i_tile+MC:
 *               for k = k_tile..k_tile+KC:
 *                   ... = A[i][k]
 *
 * After InLocalStorage(k_tile_loop, "A"):
 *   for i_tile = 0..M step MC:
 *       for k_tile = 0..K step KC:
 *           // Copy MC x KC panel
 *           for i' = 0..MC:
 *               for k' = 0..KC:
 *                   A_local[i' * KC + k'] = A[i_tile + i'][k_tile + k']
 *           for i = i_tile..i_tile+MC:
 *               for k = k_tile..k_tile+KC:
 *                   ... = A_local[(i - i_tile) * KC + (k - k_tile)]
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

    // A is a 2D array: double A[M][K] represented as Pointer to Array
    types::Array a_row(elem_desc, symbolic::symbol("K"));
    types::Pointer a_desc(a_row);
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
        k_tile_loop.root(), i, symbolic::Lt(i, symbolic::add(i_tile, MC)), i_tile, symbolic::add(i, symbolic::integer(1))
    );

    // Innermost: for k = k_tile; k < k_tile + KC; k++
    auto& k_loop = builder.add_for(
        i_loop.root(), k, symbolic::Lt(k, symbolic::add(k_tile, KC)), k_tile, symbolic::add(k, symbolic::integer(1))
    );

    // C += A[i][k]
    auto& block = builder.add_block(k_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {}, elem_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {i, k}, a_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {}, elem_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply InLocalStorage at k_tile loop level (like BLIS packing A at k_tile)
    transformations::InLocalStorage ils(k_tile_loop, "A");

    std::cout << "Testing InLocalStorage on 2D tiled panel access (BLIS-style)..." << std::endl;
    bool can_apply = ils.can_be_applied(builder_opt, am);
    std::cout << "  can_be_applied: " << (can_apply ? "yes" : "no") << std::endl;

    EXPECT_TRUE(can_apply);

    if (can_apply) {
        ils.apply(builder_opt, am);

        // Verify: local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_in_local_storage_A"));

        // Verify: k_tile_loop should now have copy loops + i_loop
        EXPECT_GE(k_tile_loop.root().size(), 2u);

        std::cout << "  Local buffer created: yes" << std::endl;
        std::cout << "  k_tile loop children after: " << k_tile_loop.root().size() << std::endl;
    }
}
