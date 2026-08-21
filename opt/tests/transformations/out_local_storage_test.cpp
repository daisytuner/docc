#include "sdfg/transformations/out_local_storage.h"

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
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/utils.h"

using namespace sdfg;

/**
 * Test: OutLocalStorage on a dynamic write (read-write accumulator)
 *
 * Before:
 *   for i = 0..4: C[i] += A[i]
 *
 * After:
 *   C_local[4]
 *   for i' = 0..4: C_local[i'] = C[i']           // init (read-write)
 *   for i = 0..4: C_local[i] += A[i]              // compute on tile
 *   for i' = 0..4: C[i'] = C_local[i']            // writeback
 */
TEST(OutLocalStorageTest, For_Array_RW) {
    builder::StructuredSDFGBuilder builder("ols_for_array_rw_test", FunctionType_CPU);

    // Create containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer c_desc(elem_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    // Create loop: for i = 0..4
    auto indvar = symbolic::symbol("i");
    auto bound = symbolic::integer(4);
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation: C[i] += A[i]
    auto& block = builder.add_block(body);
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {indvar}, c_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, c_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {indvar}, c_desc);

    // Apply transformation
    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage transformation(loop, c_out);
    EXPECT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);

    // Verify: local buffer was created
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    types::Array array_desc(elem_desc, symbolic::integer(4));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);

    // Verify: structure should now be [init_loop, main_loop, wb_loop]
    auto& new_root = builder.subject().root();
    EXPECT_EQ(new_root.size(), 3);

    // First element should be init loop
    auto* init_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(0));
    EXPECT_NE(init_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), symbolic::integer(4))));
    EXPECT_TRUE(symbolic::eq(init_loop->update(), symbolic::add(init_loop->indvar(), symbolic::integer(1))));

    auto& init_body = init_loop->root();
    EXPECT_EQ(init_body.size(), 1);
    auto* init_block = dyn_cast<structured_control_flow::Block*>(&init_body.at(0));
    EXPECT_NE(init_block, nullptr);

    EXPECT_EQ(init_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(init_block->dataflow().edges().size(), 2);
    bool reads_C = false;
    bool writes_C_local = false;
    for (auto* node : init_block->dataflow().data_nodes()) {
        if (node->data() == "C") {
            reads_C = true;
            EXPECT_EQ(init_block->dataflow().out_degree(*node), 1);
            EXPECT_EQ(init_block->dataflow().in_degree(*node), 0);

            auto& oedge = *init_block->dataflow().out_edges(*node).begin();
            EXPECT_TRUE(oedge.base_type() == c_desc);
            EXPECT_EQ(oedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), init_loop->indvar()));
        } else if (node->data() == "__daisy_out_local_storage_C0") {
            writes_C_local = true;
            EXPECT_EQ(init_block->dataflow().in_degree(*node), 1);
            EXPECT_EQ(init_block->dataflow().out_degree(*node), 0);

            auto& iedge = *init_block->dataflow().in_edges(*node).begin();
            EXPECT_TRUE(iedge.base_type() == array_desc);
            EXPECT_EQ(iedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), init_loop->indvar()));
        }
    }
    EXPECT_TRUE(reads_C);
    EXPECT_TRUE(writes_C_local);

    // Second element should be the main loop
    auto* main_loop = dyn_cast<structured_control_flow::For*>(&new_root.at(1));
    EXPECT_NE(main_loop, nullptr);

    // Verify main loop uses local buffer
    auto& main_body = main_loop->root();
    EXPECT_EQ(main_body.size(), 1);
    auto* main_block = dyn_cast<structured_control_flow::Block*>(&main_body.at(0));
    EXPECT_NE(main_block, nullptr);

    bool uses_C_local = false;
    bool uses_C_original = false;
    for (auto* node : main_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") {
            uses_C_local = true;
            for (auto& memlet : main_block->dataflow().out_edges(*node)) {
                EXPECT_TRUE(memlet.base_type() == array_desc);
                EXPECT_EQ(memlet.subset().size(), 1);
                EXPECT_TRUE(symbolic::eq(memlet.subset().at(0), main_loop->indvar()));
            }
            for (auto& memlet : main_block->dataflow().in_edges(*node)) {
                EXPECT_TRUE(memlet.base_type() == array_desc);
                EXPECT_EQ(memlet.subset().size(), 1);
                EXPECT_TRUE(symbolic::eq(memlet.subset().at(0), main_loop->indvar()));
            }
        }
        if (node->data() == "C") {
            uses_C_original = true;
        }
    }
    EXPECT_TRUE(uses_C_local);
    EXPECT_FALSE(uses_C_original);

    // Third element should be writeback loop
    auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(2));
    EXPECT_NE(wb_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), symbolic::integer(4))));
    EXPECT_TRUE(symbolic::eq(wb_loop->update(), symbolic::add(wb_loop->indvar(), symbolic::integer(1))));

    auto& wb_body = wb_loop->root();
    EXPECT_EQ(wb_body.size(), 1);
    auto* wb_block = dyn_cast<structured_control_flow::Block*>(&wb_body.at(0));
    EXPECT_NE(wb_block, nullptr);

    EXPECT_EQ(wb_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(wb_block->dataflow().edges().size(), 2);
    bool reads_C_local = false;
    bool writes_C = false;
    for (auto* node : wb_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") {
            reads_C_local = true;
            EXPECT_EQ(wb_block->dataflow().out_degree(*node), 1);
            EXPECT_EQ(wb_block->dataflow().in_degree(*node), 0);

            auto& oedge = *wb_block->dataflow().out_edges(*node).begin();
            EXPECT_TRUE(oedge.base_type() == array_desc);
            EXPECT_EQ(oedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), wb_loop->indvar()));
        } else if (node->data() == "C") {
            writes_C = true;
            EXPECT_EQ(wb_block->dataflow().in_degree(*node), 1);
            EXPECT_EQ(wb_block->dataflow().out_degree(*node), 0);

            auto& iedge = *wb_block->dataflow().in_edges(*node).begin();
            EXPECT_TRUE(iedge.base_type() == c_desc);
            EXPECT_EQ(iedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), wb_loop->indvar()));
        }
    }
    EXPECT_TRUE(reads_C_local);
    EXPECT_TRUE(writes_C);
}

/**
 * Test: OutLocalStorage on a write-only access
 *
 * Before:
 *   for i = 0..4: C[i] = A[i]
 *
 * After:
 *   C_local[4]
 *   for i = 0..4: C_local[i] = A[i]               // compute on tile, NO init
 *   for i' = 0..4: C[i'] = C_local[i']            // writeback only
 */
TEST(OutLocalStorageTest, For_Array_WO) {
    builder::StructuredSDFGBuilder builder("ols_for_array_wo_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer c_desc(elem_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // C[i] = A[i]  (write-only to C, no read)
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, c_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {indvar}, c_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage transformation(loop, c_out);
    EXPECT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);

    // Verify: local buffer was created
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    types::Array array_desc(elem_desc, symbolic::integer(4));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);

    // Verify: structure should now be [main_loop, wb_loop] — NO init
    auto& new_root = builder.subject().root();
    EXPECT_EQ(new_root.size(), 2);

    auto* main_loop = dyn_cast<structured_control_flow::For*>(&new_root.at(0));
    EXPECT_NE(main_loop, nullptr);

    // Verify main loop uses local buffer
    auto& main_body = main_loop->root();
    EXPECT_EQ(main_body.size(), 1);
    auto* main_block = dyn_cast<structured_control_flow::Block*>(&main_body.at(0));
    EXPECT_NE(main_block, nullptr);

    bool uses_C_local = false;
    bool uses_C_original = false;
    for (auto* node : main_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") {
            uses_C_local = true;
            EXPECT_EQ(main_block->dataflow().out_degree(*node), 0);
            EXPECT_EQ(main_block->dataflow().in_degree(*node), 1);

            auto& iedge = *main_block->dataflow().in_edges(*node).begin();
            EXPECT_TRUE(iedge.base_type() == array_desc);
            EXPECT_EQ(iedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), main_loop->indvar()));
        }
        if (node->data() == "C") {
            uses_C_original = true;
        }
    }
    EXPECT_TRUE(uses_C_local);
    EXPECT_FALSE(uses_C_original);

    // Second element should be writeback loop
    auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(1));
    EXPECT_NE(wb_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), symbolic::integer(4))));

    auto& wb_body = wb_loop->root();
    EXPECT_EQ(wb_body.size(), 1);
    auto* wb_block = dyn_cast<structured_control_flow::Block*>(&wb_body.at(0));
    EXPECT_NE(wb_block, nullptr);

    EXPECT_EQ(wb_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(wb_block->dataflow().edges().size(), 2);
    bool reads_C_local = false;
    bool writes_C = false;
    for (auto* node : wb_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") {
            reads_C_local = true;
            auto& oedge = *wb_block->dataflow().out_edges(*node).begin();
            EXPECT_TRUE(oedge.base_type() == array_desc);
            EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), wb_loop->indvar()));
        } else if (node->data() == "C") {
            writes_C = true;
            auto& iedge = *wb_block->dataflow().in_edges(*node).begin();
            EXPECT_TRUE(iedge.base_type() == c_desc);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), wb_loop->indvar()));
        }
    }
    EXPECT_TRUE(reads_C_local);
    EXPECT_TRUE(writes_C);
}

/**
 * Test: OutLocalStorage on a flat-pointer linearized access (read-write)
 *
 * Before:
 *   for i = 0..100: for k = 0..16: C[i*16 + k] += A[k]
 *
 * After OutLocalStorage(k_loop, C):
 *   for i = 0..100:
 *       for k' = 0..16: C_local[k'] = C[i*16 + k']    // init
 *       for k = 0..16: C_local[k] += A[k]              // compute on tile (local index k)
 *       for k' = 0..16: C[i*16 + k'] = C_local[k']    // writeback
 */
TEST(OutLocalStorageTest, For_Array_Linearized_RW) {
    builder::StructuredSDFGBuilder builder("ols_cpu_flatptr_rw", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Pointer opaque_ptr;

    builder.add_container("A", opaque_ptr, true);
    builder.add_container("C", opaque_ptr, true);
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {k}, ptr);
    // C[i*16 + k] — flat pointer linearized access, read-modify-write
    auto c_subset = symbolic::add(symbolic::mul(i, symbolic::integer(16)), k);
    builder.add_computational_memlet(block, c_in, tasklet, "_in2", {c_subset}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {c_subset}, ptr);

    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage ols(loop, c_in);
    EXPECT_TRUE(ols.can_be_applied(builder, am));
    ols.apply(builder, am);

    // Verify: buffer created
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    types::Array array_desc(elem, symbolic::integer(16));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);

    // Outer loop preserved at root
    EXPECT_EQ(builder.subject().root().size(), 1);

    // Structure inside outer loop = [init_map, main_loop, wb_map]
    auto& outer_body = outer_loop.root();
    EXPECT_EQ(outer_body.size(), 3u);

    auto* init_map = dyn_cast<structured_control_flow::Map*>(&outer_body.at(0));
    EXPECT_NE(init_map, nullptr);
    EXPECT_TRUE(symbolic::eq(init_map->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_map->condition(), symbolic::Lt(init_map->indvar(), symbolic::integer(16))));

    auto& init_body = init_map->root();
    EXPECT_EQ(init_body.size(), 1);
    auto* init_block = dyn_cast<structured_control_flow::Block*>(&init_body.at(0));
    EXPECT_NE(init_block, nullptr);

    EXPECT_EQ(init_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(init_block->dataflow().edges().size(), 2);
    for (auto* node : init_block->dataflow().data_nodes()) {
        if (node->data() == "C") {
            auto& oedge = *init_block->dataflow().out_edges(*node).begin();
            EXPECT_TRUE(oedge.base_type() == ptr);
            EXPECT_EQ(oedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(
                oedge.subset().at(0),
                symbolic::add(symbolic::mul(outer_loop.indvar(), symbolic::integer(16)), init_map->indvar())
            ));
        } else if (node->data() == "__daisy_out_local_storage_C0") {
            auto& iedge = *init_block->dataflow().in_edges(*node).begin();
            EXPECT_TRUE(iedge.base_type() == array_desc);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), init_map->indvar()));
        }
    }

    auto* main_loop = dyn_cast<structured_control_flow::For*>(&outer_body.at(1));
    EXPECT_NE(main_loop, nullptr);

    auto* wb_map = dyn_cast<structured_control_flow::Map*>(&outer_body.at(2));
    EXPECT_NE(wb_map, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_map->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_map->condition(), symbolic::Lt(wb_map->indvar(), symbolic::integer(16))));

    // Verify the compute memlets use LOCAL indices (k, zero-based)
    auto& main_body = main_loop->root();
    EXPECT_EQ(main_body.size(), 1u);
    auto* compute_block = dyn_cast<structured_control_flow::Block*>(&main_body.at(0));
    EXPECT_NE(compute_block, nullptr);

    bool found_local_access = false;
    for (auto* node : compute_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") {
            found_local_access = true;
            for (auto& memlet : compute_block->dataflow().out_edges(*node)) {
                EXPECT_EQ(memlet.subset().size(), 1u);
                EXPECT_TRUE(symbolic::eq(memlet.subset().at(0), k));
            }
            for (auto& memlet : compute_block->dataflow().in_edges(*node)) {
                EXPECT_EQ(memlet.subset().size(), 1u);
                EXPECT_TRUE(symbolic::eq(memlet.subset().at(0), k));
            }
        }
    }
    EXPECT_TRUE(found_local_access);
}

/**
 * Test: OutLocalStorage CPU_Stack with flat pointer (non-GPU baseline)
 *
 * Setup: for i = 0..N: for k = 0..16: C[i*16 + k] = A[k]
 * After: for loop writes local[k], then Map(0..16) copies local[d] → C[i*16+d]
 */
TEST(OutLocalStorageTest, For_Array_Linearized_WO) {
    builder::StructuredSDFGBuilder builder("ols_cpu_flatptr", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::UInt64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr, true);
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
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {k}, ptr);
    // C[i*16 + k] — flat pointer linearized access
    builder.add_computational_memlet(
        block, tasklet, "_out", c_out, {symbolic::add(symbolic::mul(i, symbolic::integer(16)), k)}, ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage ols(loop, c_out);
    EXPECT_TRUE(ols.can_be_applied(builder_opt, am));
    ols.apply(builder_opt, am);


    // Verify: buffer created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

    // Structure inside outer loop = [main_loop, writeback_map]
    auto& outer_body = outer_loop.root();
    EXPECT_EQ(outer_body.size(), 2u);

    auto* main_loop = dyn_cast<structured_control_flow::For*>(&outer_body.at(0));
    EXPECT_NE(main_loop, nullptr);

    auto* wb_map = dyn_cast<structured_control_flow::Map*>(&outer_body.at(1));
    EXPECT_NE(wb_map, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_map->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_map->condition(), symbolic::Lt(wb_map->indvar(), symbolic::integer(16))));

    // Verify the compute memlet uses LOCAL indices (k, zero-based)
    auto& main_body = main_loop->root();
    EXPECT_EQ(main_body.size(), 1u);
    auto* compute_block = dyn_cast<structured_control_flow::Block*>(&main_body.at(0));
    EXPECT_NE(compute_block, nullptr);

    bool found_local_access = false;
    for (auto* node : compute_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") {
            // Check incoming memlets (writes to this access node)
            for (auto& memlet : compute_block->dataflow().in_edges(*node)) {
                found_local_access = true;
                // After OLS, the subset should be {k} (local index, zero-based)
                auto& subset = memlet.subset();
                EXPECT_EQ(subset.size(), 1u);
                EXPECT_TRUE(symbolic::eq(subset.at(0), k));
            }
        }
    }
    EXPECT_TRUE(found_local_access);
}

/**
 * Test: OutLocalStorage on a PolyBench-style 2D nested array pointer (write-only)
 *
 * Before:
 *   for i = 0..4: for j = 0..8: C[0][i][j] = A[0][i][j]
 *
 * After OutLocalStorage(outer_loop, C):
 *   C_local[32] (flattened, the leading dim is extent 1)
 *   for i = 0..4: for j = 0..8: C_local[i*8 + j] = A[0][i][j]    // compute
 *   for d0 = 0..4: for d1 = 0..8: C[0][d0][d1] = C_local[...]    // writeback
 */
TEST(OutLocalStorageTest, For_Array_PolyBench_WO) {
    builder::StructuredSDFGBuilder builder("ols_polybench_wo", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array array_desc(elem_desc, symbolic::integer(8));
    types::Array array_desc_2d(array_desc, symbolic::integer(4));
    types::Pointer ptr_desc(array_desc_2d);
    types::Pointer flat_ptr_desc(elem_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // C[0][i][j] = A[0][i][j]
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::integer(0), i, j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::integer(0), i, j}, ptr_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage transformation(outer_loop, c_out);
    EXPECT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);

    // Verify: local buffer was created (flattened to 32)
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    types::Array array_desc_ref(elem_desc, symbolic::integer(32));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc_ref);

    // Verify: structure should be [main_loop, wb_loop_outer] — NO init
    auto& new_root = builder.subject().root();
    EXPECT_EQ(new_root.size(), 2);

    auto* main_loop = dyn_cast<structured_control_flow::For*>(&new_root.at(0));
    EXPECT_NE(main_loop, nullptr);

    auto* wb_outer = dyn_cast<structured_control_flow::Map*>(&new_root.at(1));
    EXPECT_NE(wb_outer, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_outer->condition(), symbolic::Lt(wb_outer->indvar(), symbolic::integer(4))));

    auto* wb_inner = dyn_cast<structured_control_flow::Map*>(&wb_outer->root().at(0));
    EXPECT_NE(wb_inner, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_inner->condition(), symbolic::Lt(wb_inner->indvar(), symbolic::integer(8))));

    auto* wb_block = dyn_cast<structured_control_flow::Block*>(&wb_inner->root().at(0));
    EXPECT_NE(wb_block, nullptr);

    EXPECT_EQ(wb_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(wb_block->dataflow().edges().size(), 2);
    for (auto* node : wb_block->dataflow().data_nodes()) {
        if (node->data() == "C") {
            auto& iedge = *wb_block->dataflow().in_edges(*node).begin();
            EXPECT_TRUE(iedge.base_type() == flat_ptr_desc);
            EXPECT_EQ(iedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(
                iedge.subset().at(0),
                symbolic::add(wb_inner->indvar(), symbolic::mul(wb_outer->indvar(), symbolic::integer(8)))
            ));
        } else if (node->data() == "__daisy_out_local_storage_C0") {
            auto& oedge = *wb_block->dataflow().out_edges(*node).begin();
            EXPECT_TRUE(oedge.base_type() == array_desc_ref);
            EXPECT_EQ(oedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(
                oedge.subset().at(0),
                symbolic::add(wb_inner->indvar(), symbolic::mul(wb_outer->indvar(), symbolic::integer(8)))
            ));
        }
    }

    // Verify main loop uses local buffer
    bool uses_C_local = false;
    bool uses_C_original = false;
    auto* main_inner = dyn_cast<structured_control_flow::For*>(&main_loop->root().at(0));
    EXPECT_NE(main_inner, nullptr);
    auto* main_block = dyn_cast<structured_control_flow::Block*>(&main_inner->root().at(0));
    EXPECT_NE(main_block, nullptr);
    for (auto* node : main_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") {
            uses_C_local = true;
        }
        if (node->data() == "C") {
            uses_C_original = true;
        }
    }
    EXPECT_TRUE(uses_C_local);
    EXPECT_FALSE(uses_C_original);
}

TEST(OutLocalStorageTest, For_Array_PolyBench_RW) {
    builder::StructuredSDFGBuilder builder("ols_flat_2d", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Array array_desc(elem_desc, symbolic::integer(4));
    types::Array array_desc_2d(array_desc, symbolic::integer(8));
    types::Pointer ptr_desc(array_desc_2d);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // Outer loop: for i = 0..4
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    // Inner loop: for j = 0..8
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // C[0][i][j] += A[0][i][j]
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {symbolic::integer(0), i, j}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::integer(0), i, j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::integer(0), i, j}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage transformation(outer_loop, c_in);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    builder_opt.subject().validate();

    // Verify local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

    // Structure: [init_loop(s), outer_loop, writeback_loop(s)]
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 3);

    // Init should be a for loop (first dimension)
    auto* init_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(0));
    EXPECT_NE(init_loop, nullptr);
    // Should iterate 0..4 (first dim extent)
    EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), symbolic::integer(4))));

    // Init loop should contain nested loop for second dimension
    auto& init_body = init_loop->root();
    EXPECT_EQ(init_body.size(), 1);
    auto* inner_init = dyn_cast<structured_control_flow::Map*>(&init_body.at(0));
    EXPECT_NE(inner_init, nullptr);
    EXPECT_TRUE(symbolic::eq(inner_init->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(inner_init->condition(), symbolic::Lt(inner_init->indvar(), symbolic::integer(8))));

    // check that accesses got converted into linearized accesses
    auto* inner_init_body = dyn_cast<structured_control_flow::Block*>(&inner_init->root().at(0));
    EXPECT_NE(inner_init_body, nullptr);
    for (auto& edge : inner_init_body->dataflow().edges()) {
        auto inferred_type = types::infer_type(builder_opt.subject(), edge.base_type(), edge.subset());
        EXPECT_TRUE(inferred_type->type_id() == types::TypeID::Scalar);
        EXPECT_EQ(inferred_type->primitive_type(), types::PrimitiveType::Double);
    }

    // Compute loop preserved
    auto* compute_loop = dyn_cast<structured_control_flow::For*>(&new_root.at(1));
    EXPECT_NE(compute_loop, nullptr);

    // Writeback should be a for loop
    auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(2));
    EXPECT_NE(wb_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), symbolic::integer(4))));

    auto& wb_body = wb_loop->root();
    EXPECT_EQ(wb_body.size(), 1);
    auto* inner_wb = dyn_cast<structured_control_flow::Map*>(&wb_body.at(0));
    EXPECT_NE(inner_wb, nullptr);
    EXPECT_TRUE(symbolic::eq(inner_wb->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(inner_wb->condition(), symbolic::Lt(inner_wb->indvar(), symbolic::integer(8))));

    auto* inner_wb_body = dyn_cast<structured_control_flow::Block*>(&inner_wb->root().at(0));
    EXPECT_NE(inner_wb_body, nullptr);
    for (auto& edge : inner_wb_body->dataflow().edges()) {
        auto inferred_type = types::infer_type(builder_opt.subject(), edge.base_type(), edge.subset());
        EXPECT_TRUE(inferred_type->type_id() == types::TypeID::Scalar);
    }
}

/**
 * Test: OutLocalStorage on a scalar (constant index) access (read-write)
 *
 * Before:
 *   for i = 0..4: C[0] += A[i]
 *
 * After:
 *   C_local[1]
 *   init_block:   C_local[0] = C[0]               // init (read-write, scalar)
 *   for i = 0..4: C_local[0] += A[i]              // compute on tile
 *   wb_block:     C[0] = C_local[0]               // writeback
 */
TEST(OutLocalStorageTest, For_Scalar_RW) {
    builder::StructuredSDFGBuilder builder("ols_for_scalar_rw_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer c_desc(elem_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // C[0] += A[i]
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {symbolic::integer(0)}, c_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, c_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::integer(0)}, c_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage transformation(loop, c_out);
    EXPECT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);

    // Verify: local buffer was created (size 1)
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    types::Array array_desc(elem_desc, symbolic::integer(1));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);

    // Verify: structure should now be [init_block, main_loop, wb_block]
    auto& new_root = builder.subject().root();
    EXPECT_EQ(new_root.size(), 3);

    // First element should be init block (no loop because extent is 1)
    auto* init_block = dyn_cast<structured_control_flow::Block*>(&new_root.at(0));
    EXPECT_NE(init_block, nullptr);

    EXPECT_EQ(init_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(init_block->dataflow().edges().size(), 2);
    bool reads_C = false;
    bool writes_C_local = false;
    for (auto* node : init_block->dataflow().data_nodes()) {
        if (node->data() == "C") {
            reads_C = true;
            auto& oedge = *init_block->dataflow().out_edges(*node).begin();
            EXPECT_TRUE(oedge.base_type() == c_desc);
            EXPECT_EQ(oedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), symbolic::zero()));
        } else if (node->data() == "__daisy_out_local_storage_C0") {
            writes_C_local = true;
            auto& iedge = *init_block->dataflow().in_edges(*node).begin();
            EXPECT_TRUE(iedge.base_type() == array_desc);
            EXPECT_EQ(iedge.subset().size(), 1);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), symbolic::zero()));
        }
    }
    EXPECT_TRUE(reads_C);
    EXPECT_TRUE(writes_C_local);

    // Second element should be the main loop
    auto* main_loop = dyn_cast<structured_control_flow::For*>(&new_root.at(1));
    EXPECT_NE(main_loop, nullptr);

    auto& main_body = main_loop->root();
    auto* main_block = dyn_cast<structured_control_flow::Block*>(&main_body.at(0));
    EXPECT_NE(main_block, nullptr);
    bool uses_C_local = false;
    bool uses_C_original = false;
    for (auto* node : main_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") {
            uses_C_local = true;
            for (auto& m : main_block->dataflow().out_edges(*node)) {
                EXPECT_TRUE(m.base_type() == array_desc);
                EXPECT_TRUE(symbolic::eq(m.subset().at(0), symbolic::zero()));
            }
            for (auto& m : main_block->dataflow().in_edges(*node)) {
                EXPECT_TRUE(m.base_type() == array_desc);
                EXPECT_TRUE(symbolic::eq(m.subset().at(0), symbolic::zero()));
            }
        }
        if (node->data() == "C") uses_C_original = true;
    }
    EXPECT_TRUE(uses_C_local);
    EXPECT_FALSE(uses_C_original);

    // Third element should be writeback block (no loop)
    auto* wb_block = dyn_cast<structured_control_flow::Block*>(&new_root.at(2));
    EXPECT_NE(wb_block, nullptr);

    EXPECT_EQ(wb_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(wb_block->dataflow().edges().size(), 2);
    bool reads_C_local = false;
    bool writes_C = false;
    for (auto* node : wb_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") {
            reads_C_local = true;
            auto& oedge = *wb_block->dataflow().out_edges(*node).begin();
            EXPECT_TRUE(oedge.base_type() == array_desc);
            EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), symbolic::zero()));
        } else if (node->data() == "C") {
            writes_C = true;
            auto& iedge = *wb_block->dataflow().in_edges(*node).begin();
            EXPECT_TRUE(iedge.base_type() == c_desc);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), symbolic::zero()));
        }
    }
    EXPECT_TRUE(reads_C_local);
    EXPECT_TRUE(writes_C);
}

/**
 * Test: OutLocalStorage on a scalar (constant index) access (write-only)
 *
 * Before:
 *   for i = 0..4: C[0] = A[i]
 *
 * After:
 *   C_local[1]
 *   for i = 0..4: C_local[0] = A[i]               // compute on tile, NO init
 *   wb_block:     C[0] = C_local[0]               // writeback
 */
TEST(OutLocalStorageTest, For_Scalar_WO) {
    builder::StructuredSDFGBuilder builder("ols_for_scalar_wo_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer c_desc(elem_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1))
    );

    // C[0] = A[i] (last-write-wins write-only pattern)
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, c_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::integer(0)}, c_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage transformation(loop, c_out);
    EXPECT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);

    // Verify: local buffer was created (size 1)
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    types::Array array_desc(elem_desc, symbolic::integer(1));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);

    // Verify: structure should be [main_loop, wb_block] — NO init
    auto& new_root = builder.subject().root();
    EXPECT_EQ(new_root.size(), 2);

    auto* main_loop = dyn_cast<structured_control_flow::For*>(&new_root.at(0));
    EXPECT_NE(main_loop, nullptr);

    auto* wb_block = dyn_cast<structured_control_flow::Block*>(&new_root.at(1));
    EXPECT_NE(wb_block, nullptr);

    EXPECT_EQ(wb_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(wb_block->dataflow().edges().size(), 2);
    for (auto* node : wb_block->dataflow().data_nodes()) {
        if (node->data() == "C") {
            auto& iedge = *wb_block->dataflow().in_edges(*node).begin();
            EXPECT_TRUE(iedge.base_type() == c_desc);
            EXPECT_TRUE(symbolic::eq(iedge.subset().at(0), symbolic::zero()));
        } else if (node->data() == "__daisy_out_local_storage_C0") {
            auto& oedge = *wb_block->dataflow().out_edges(*node).begin();
            EXPECT_TRUE(oedge.base_type() == array_desc);
            EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), symbolic::zero()));
        }
    }

    // Verify main loop uses local buffer
    auto* main_block = dyn_cast<structured_control_flow::Block*>(&main_loop->root().at(0));
    EXPECT_NE(main_block, nullptr);
    bool uses_C_local = false;
    bool uses_C_original = false;
    for (auto* node : main_block->dataflow().data_nodes()) {
        if (node->data() == "__daisy_out_local_storage_C0") uses_C_local = true;
        if (node->data() == "C") uses_C_original = true;
    }
    EXPECT_TRUE(uses_C_local);
    EXPECT_FALSE(uses_C_original);
}

/**
 * Test: OutLocalStorage on a Map loop (read-write)
 *
 * Same as For_Array_RW but with `Map` instead of `For`.
 */
TEST(OutLocalStorageTest, Map_Array_RW) {
    builder::StructuredSDFGBuilder builder("ols_map_array_rw_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer c_desc(elem_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {indvar}, c_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, c_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {indvar}, c_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage transformation(loop, c_out);
    EXPECT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);

    // Verify: local buffer was created
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    types::Array array_desc(elem_desc, symbolic::integer(4));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);

    // Verify: structure should now be [init_loop, main_loop, wb_loop]
    auto& new_root = builder.subject().root();
    EXPECT_EQ(new_root.size(), 3);

    auto* init_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(0));
    EXPECT_NE(init_loop, nullptr);

    auto* main_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(1));
    EXPECT_NE(main_loop, nullptr);

    auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(2));
    EXPECT_NE(wb_loop, nullptr);
}

/**
 * Test: OutLocalStorage on a Map loop (write-only)
 *
 * Same as For_Array_WO but with `Map` instead of `For`.
 */
TEST(OutLocalStorageTest, Map_Array_WO) {
    builder::StructuredSDFGBuilder builder("ols_map_array_wo_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer c_desc(elem_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, c_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {indvar}, c_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage transformation(loop, c_out);
    EXPECT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);

    // Verify: local buffer was created
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    types::Array array_desc(elem_desc, symbolic::integer(4));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);

    // Verify: structure should be [main_loop, wb_loop] — NO init
    auto& new_root = builder.subject().root();
    EXPECT_EQ(new_root.size(), 2);

    auto* main_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(0));
    EXPECT_NE(main_loop, nullptr);

    auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(1));
    EXPECT_NE(wb_loop, nullptr);
}

/**
 * Test: OutLocalStorage applied twice for two disjoint write groups of the same container
 *
 * Block 1 writes to the lower half of C: row i in [0, N)        → C[i*K + k]
 * Block 2 writes to the upper half of C: row (N+j) in [N, 2N)   → C[(N+j)*K + k]
 *
 * The two regions are convex and provably disjoint for all (i, j), so packing
 * each group into its own local buffer is semantically equivalent to the
 * original program.
 */
TEST(OutLocalStorageTest, For_MultipleGroups_RW) {
    builder::StructuredSDFGBuilder builder("ols_multi_groups_rw_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    types::Pointer opaque_desc;

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto K = symbolic::integer(16);

    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    auto& j_loop =
        builder.add_for(i_loop.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::one()));
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // Block 1: C[i*16 + k] += A[k]
    auto& block1 = builder.add_block(k_loop.root());
    auto& cik_in = builder.add_access(block1, "C");
    auto& cik_out = builder.add_access(block1, "C");
    auto& a1_in = builder.add_access(block1, "A");
    auto& t1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_c", "_a"});
    auto lin_ik = symbolic::add(symbolic::mul(i, K), k);
    builder.add_computational_memlet(block1, cik_in, t1, "_c", {lin_ik}, ptr_desc);
    builder.add_computational_memlet(block1, a1_in, t1, "_a", {k}, ptr_desc);
    builder.add_computational_memlet(block1, t1, "_out", cik_out, {lin_ik}, ptr_desc);

    // Block 2: C[(N+j)*16 + k] += A[k]  — upper half, disjoint from block 1
    auto& block2 = builder.add_block(k_loop.root());
    auto& cjk_in = builder.add_access(block2, "C");
    auto& cjk_out = builder.add_access(block2, "C");
    auto& a2_in = builder.add_access(block2, "A");
    auto& t2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_c", "_a"});
    auto lin_jk = symbolic::add(symbolic::mul(symbolic::add(N, j), K), k);
    builder.add_computational_memlet(block2, cjk_in, t2, "_c", {lin_jk}, ptr_desc);
    builder.add_computational_memlet(block2, a2_in, t2, "_a", {k}, ptr_desc);
    builder.add_computational_memlet(block2, t2, "_out", cjk_out, {lin_jk}, ptr_desc);

    // First OLS: pack C[i,k] group
    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage ols_ik(k_loop, cik_out);
    ASSERT_TRUE(ols_ik.can_be_applied(builder, am));
    ols_ik.apply(builder, am);

    // Second OLS: pack C[j,k] group
    transformations::OutLocalStorage ols_jk(k_loop, cjk_out);
    EXPECT_TRUE(ols_jk.can_be_applied(builder, am));
    ols_jk.apply(builder, am);

    // Verify: two local buffers were created
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C1"));
    types::Array array_desc(elem_desc, symbolic::integer(16));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C1") == array_desc);

    // No raw C accesses should remain inside the k-loop
    auto& k_body = k_loop.root();
    for (size_t bi = 0; bi < k_body.size(); ++bi) {
        auto* blk = dyn_cast<structured_control_flow::Block*>(&k_body.at(bi));
        if (!blk) continue;
        for (auto* node : blk->dataflow().data_nodes()) {
            EXPECT_NE(node->data(), "C") << "All C accesses inside k-loop should be rewritten";
        }
    }
}

/**
 * Test: OutLocalStorage applied twice for two disjoint write groups (write-only)
 *
 * Same disjoint lower-half/upper-half pattern as For_MultipleGroups_RW, but the
 * tasklets are pure assignments (no read of C) so OLS takes the write-only path
 * and emits no init copy.
 */
TEST(OutLocalStorageTest, For_MultipleGroups_WO) {
    builder::StructuredSDFGBuilder builder("ols_multi_groups_wo_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    types::Pointer opaque_desc;

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto K = symbolic::integer(16);

    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    auto& j_loop =
        builder.add_for(i_loop.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::one()));
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // Block 1: C[i*16 + k] = A[k]
    auto& block1 = builder.add_block(k_loop.root());
    auto& cik_out = builder.add_access(block1, "C");
    auto& a1_in = builder.add_access(block1, "A");
    auto& t1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_a"});
    auto lin_ik = symbolic::add(symbolic::mul(i, K), k);
    builder.add_computational_memlet(block1, a1_in, t1, "_a", {k}, ptr_desc);
    builder.add_computational_memlet(block1, t1, "_out", cik_out, {lin_ik}, ptr_desc);

    // Block 2: C[(N+j)*16 + k] = A[k]  — upper half, disjoint from block 1
    auto& block2 = builder.add_block(k_loop.root());
    auto& cjk_out = builder.add_access(block2, "C");
    auto& a2_in = builder.add_access(block2, "A");
    auto& t2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_a"});
    auto lin_jk = symbolic::add(symbolic::mul(symbolic::add(N, j), K), k);
    builder.add_computational_memlet(block2, a2_in, t2, "_a", {k}, ptr_desc);
    builder.add_computational_memlet(block2, t2, "_out", cjk_out, {lin_jk}, ptr_desc);

    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage ols_ik(k_loop, cik_out);
    ASSERT_TRUE(ols_ik.can_be_applied(builder, am));
    ols_ik.apply(builder, am);

    transformations::OutLocalStorage ols_jk(k_loop, cjk_out);
    EXPECT_TRUE(ols_jk.can_be_applied(builder, am));
    ols_jk.apply(builder, am);

    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C1"));
    types::Array array_desc(elem_desc, symbolic::integer(16));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C1") == array_desc);

    // No raw C accesses should remain inside the k-loop
    auto& k_body = k_loop.root();
    for (size_t bi = 0; bi < k_body.size(); ++bi) {
        auto* blk = dyn_cast<structured_control_flow::Block*>(&k_body.at(bi));
        if (!blk) continue;
        for (auto* node : blk->dataflow().data_nodes()) {
            EXPECT_NE(node->data(), "C") << "All C accesses inside k-loop should be rewritten";
        }
    }
}

/**
 * Test: OutLocalStorage on a single access node with multiple write memlets (write-only)
 *
 * A single c_out access node receives two writes from two tasklets at disjoint
 * subsets C[i*K+k] (lower half) and C[(N+j)*K+k] (upper half). After the first
 * OLS, the access node must be split: one group rewritten to a new local access
 * node, the other untouched. The disjoint regions make this transformation
 * semantically safe.
 */
TEST(OutLocalStorageTest, For_MultipleGroups_SplitNode_WO) {
    builder::StructuredSDFGBuilder builder("ols_split_node_wo_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    types::Pointer opaque_desc;

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    auto& root = builder.subject().root();

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto K = symbolic::integer(16);

    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));
    auto& j_loop =
        builder.add_for(i_loop.root(), j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::one()));
    auto& k_loop =
        builder.add_for(j_loop.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::one()));

    // Single block with one shared c_out access node receiving two writes
    auto& block = builder.add_block(k_loop.root());
    auto& c_out = builder.add_access(block, "C");
    auto& a1_in = builder.add_access(block, "A");
    auto& a2_in = builder.add_access(block, "A");
    auto& t1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_a"});
    auto& t2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_a"});

    auto lin_ik = symbolic::add(symbolic::mul(i, K), k);
    auto lin_jk = symbolic::add(symbolic::mul(symbolic::add(N, j), K), k);

    builder.add_computational_memlet(block, a1_in, t1, "_a", {k}, ptr_desc);
    builder.add_computational_memlet(block, t1, "_out", c_out, {lin_ik}, ptr_desc);
    builder.add_computational_memlet(block, a2_in, t2, "_a", {k}, ptr_desc);
    builder.add_computational_memlet(block, t2, "_out", c_out, {lin_jk}, ptr_desc);

    // First OLS: pack C[i,k] group
    analysis::AnalysisManager am(builder.subject());
    transformations::OutLocalStorage ols_ik(k_loop, c_out);
    ASSERT_TRUE(ols_ik.can_be_applied(builder, am));
    ols_ik.apply(builder, am);

    // Find the remaining "C" access node (should be exactly one after split)
    const data_flow::AccessNode* new_c_out = nullptr;
    for (auto* node : block.dataflow().data_nodes()) {
        if (node->data() == "C") {
            EXPECT_TRUE(new_c_out == nullptr); // should only be one access to C after first OLS
            new_c_out = node;
        }
    }
    EXPECT_NE(new_c_out, nullptr);

    // Second OLS: pack C[j,k] group via the remaining access node
    transformations::OutLocalStorage ols_jk(k_loop, *new_c_out);
    EXPECT_TRUE(ols_jk.can_be_applied(builder, am));
    ols_jk.apply(builder, am);

    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C0"));
    EXPECT_TRUE(builder.subject().exists("__daisy_out_local_storage_C1"));
    types::Array array_desc(elem_desc, symbolic::integer(16));
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C0") == array_desc);
    EXPECT_TRUE(builder.subject().type("__daisy_out_local_storage_C1") == array_desc);

    // No raw C accesses should remain inside the k-loop
    auto& k_body = k_loop.root();
    for (size_t bi = 0; bi < k_body.size(); ++bi) {
        auto* blk = dyn_cast<structured_control_flow::Block*>(&k_body.at(bi));
        if (!blk) continue;
        for (auto* node : blk->dataflow().data_nodes()) {
            EXPECT_NE(node->data(), "C") << "All C accesses inside k-loop should be rewritten";
        }
    }
}

/**
 * Test: OutLocalStorage should fail when container is not used in the loop
 */
TEST(OutLocalStorageTest, FailsOnUnusedContainer) {
    builder::StructuredSDFGBuilder builder("ols_unused_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc, true); // declared but not used inside loop

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

    // Only use A inside the loop (write so OLS otherwise would apply)
    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, ptr_desc);

    analysis::AnalysisManager am(builder.subject());

    // OLS should FAIL on B (not used in loop body)
    transformations::OutLocalStorage ols(loop, b_outside);
    EXPECT_FALSE(ols.can_be_applied(builder, am));
}

/**
 * Test: JSON serialization round-trip
 */
TEST(OutLocalStorageTest, JsonSerialization) {
    builder::StructuredSDFGBuilder builder("ols_json_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer c_desc(elem_desc);
    builder.add_container("A", c_desc, true);
    builder.add_container("C", c_desc, true);

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
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {indvar}, c_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {indvar}, c_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {indvar}, c_desc);

    analysis::AnalysisManager am(builder.subject());

    transformations::OutLocalStorage original(loop, c_out);
    EXPECT_TRUE(original.can_be_applied(builder, am));

    nlohmann::json j;
    original.to_json(j);

    EXPECT_EQ(j["transformation_type"], "OutLocalStorage");
    EXPECT_TRUE(j.contains("subgraph"));

    // Deserialize and verify
    auto deserialized = transformations::OutLocalStorage::from_json(builder, j);
    EXPECT_EQ(deserialized.name(), "OutLocalStorage");
    EXPECT_TRUE(deserialized.can_be_applied(builder, am));
}


/**
 * Test: OutLocalStorage fails on read-only container
 *
 * for i = 0..4:
 *     for j = 0..8:
 *         B[j] = A[j]   (A is read-only, not written)
 */
TEST(OutLocalStorageTest, FailsOnReadOnly) {
    builder::StructuredSDFGBuilder builder("ols_ro_fail", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc);

    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    auto j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // B[j] = A[j] — A is only read
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& b_out = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", b_out, {j}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // A is read-only — should fail
    transformations::OutLocalStorage transformation(outer_loop, a_in);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, am));
}

/**
 * Test: OutLocalStorage fails on access node outside the loop
 */
TEST(OutLocalStorageTest, FailsOnAccessOutsideLoop) {
    builder::StructuredSDFGBuilder builder("ols_outside_fail", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);
    builder.add_container("B", ptr_desc);

    auto& root = builder.subject().root();

    // Place an access to B outside the loop
    auto& outer_block = builder.add_block(root);
    auto& b_outside = builder.add_access(outer_block, "B");
    auto& i_outside = builder.add_access(outer_block, "i");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(outer_block, i_outside, tasklet_outside, "_in", {});
    builder.add_computational_memlet(outer_block, tasklet_outside, "_out", b_outside, {symbolic::integer(0)}, ptr_desc);

    auto i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    auto j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // C[j] += A[j] inside the loop
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {j}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {j}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // b_outside is not associated with loop body — should fail
    transformations::OutLocalStorage transformation(outer_loop, b_outside);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, am));
}

/**
 * Test: OutLocalStorage with flat pointer and linearized 2D access
 *
 * Before:
 *   for i = 0..4:
 *       for j = 0..8:
 *           C[i*8+j] += A[i*8+j]
 *
 * After OutLocalStorage(i_loop, C):
 *   for d0 = 0..4:
 *       for d1 = 0..8:
 *           C_local[d0*8+d1] = C[d0*8+d1]
 *   for i = 0..4:
 *       for j = 0..8:
 *           C_local[(i-0)*8+(j-0)] += A[i*8+j]
 *   for d0 = 0..4:
 *       for d1 = 0..8:
 *           C[d0*8+d1] = C_local[d0*8+d1]
 *
 * Buffer size: 4 * 8 = 32 (linearized flat pointer)
 */
TEST(OutLocalStorageTest, FlatPointer_Linearized2D) {
    builder::StructuredSDFGBuilder builder("ols_flat_2d", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // Outer loop: for i = 0..4
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    // Inner loop: for j = 0..8
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // C[i*8+j] += A[i*8+j]
    auto linear_idx = symbolic::add(symbolic::mul(i, symbolic::integer(8)), j);
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {linear_idx}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {linear_idx}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {linear_idx}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage transformation(outer_loop, c_in);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);


    // Verify local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

    // Structure: [init_loop(s), outer_loop, writeback_loop(s)]
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 3);

    // Init should be a for loop (first dimension)
    auto* init_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(0));
    EXPECT_NE(init_loop, nullptr);
    // Should iterate 0..4 (first dim extent)
    EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), symbolic::integer(4))));

    // Init loop should contain nested loop for second dimension
    auto& init_body = init_loop->root();
    EXPECT_EQ(init_body.size(), 1);
    auto* inner_init = dyn_cast<structured_control_flow::Map*>(&init_body.at(0));
    EXPECT_NE(inner_init, nullptr);
    EXPECT_TRUE(symbolic::eq(inner_init->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(inner_init->condition(), symbolic::Lt(inner_init->indvar(), symbolic::integer(8))));

    // Compute loop preserved
    auto* compute_loop = dyn_cast<structured_control_flow::For*>(&new_root.at(1));
    EXPECT_NE(compute_loop, nullptr);

    // Writeback should be a for loop
    auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&new_root.at(2));
    EXPECT_NE(wb_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), symbolic::integer(4))));
}

/**
 * Test: OutLocalStorage with tiled loop and symbolic bounds
 *
 * Before:
 *   for i_tile = 0..N step MC:
 *       for k_tile = 0..K step KC:
 *           for i = i_tile..min(i_tile+MC, N):
 *               for k = k_tile..min(k_tile+KC, K):
 *                   C[i*K+k] += A[i*K+k]
 *
 * After OutLocalStorage(i_loop, C):
 *   The tile extents at i_loop level are MC x KC (constant after overapprox):
 *   for i_tile = 0..N step MC:
 *       for k_tile = 0..K step KC:
 *           // Init: copy MC*KC tile from C
 *           for d0 = 0..MC:
 *               for d1 = 0..KC:
 *                   C_local[d0*KC+d1] = C[linearize(i_tile+d0, k_tile+d1)]
 *           // Compute
 *           for i = i_tile..min(i_tile+MC, N):
 *               for k = k_tile..min(k_tile+KC, K):
 *                   C_local[...] += A[i*K+k]
 *           // Writeback
 *           for d0 = 0..MC:
 *               for d1 = 0..KC:
 *                   C[linearize(i_tile+d0, k_tile+d1)] = C_local[d0*KC+d1]
 *
 * Buffer size: MC * KC (constant, known at compile time)
 */
TEST(OutLocalStorageTest, TiledAccumulator_2D) {
    builder::StructuredSDFGBuilder builder("ols_tiled_2d", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("k_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("k", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto MC = symbolic::integer(64);
    auto KC = symbolic::integer(128);
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    auto i_tile = symbolic::symbol("i_tile");
    auto k_tile = symbolic::symbol("k_tile");
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");

    // for i_tile = 0; i_tile < N; i_tile += MC
    auto& i_tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, MC));

    // for k_tile = 0; k_tile < K; k_tile += KC
    auto& k_tile_loop =
        builder
            .add_for(i_tile_loop.root(), k_tile, symbolic::Lt(k_tile, K), symbolic::integer(0), symbolic::add(k_tile, KC));

    // for i = i_tile; i < min(i_tile+MC, N); i++
    auto& i_loop = builder.add_for(
        k_tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, MC)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // for k = k_tile; k < min(k_tile+KC, K); k++
    auto& k_loop = builder.add_for(
        i_loop.root(),
        k,
        symbolic::And(symbolic::Lt(k, symbolic::add(k_tile, KC)), symbolic::Lt(k, K)),
        k_tile,
        symbolic::add(k, symbolic::one())
    );

    // C[i*K+k] += A[i*K+k]
    auto linear_idx = symbolic::add(symbolic::mul(i, K), k);
    auto& block = builder.add_block(k_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {linear_idx}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {linear_idx}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {linear_idx}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply OutLocalStorage at i_loop level (tile at this level has extents MC x KC)
    transformations::OutLocalStorage transformation(i_loop, c_in);

    bool can_apply = transformation.can_be_applied(builder_opt, am);
    EXPECT_TRUE(can_apply);

    if (can_apply) {
        transformation.apply(builder_opt, am);


        // Verify local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

        // Structure: k_tile_loop body should be [init_loops, i_loop, writeback_loops]
        auto& k_tile_body = k_tile_loop.root();
        EXPECT_EQ(k_tile_body.size(), 3);

        // Init: nested for loops (MC x KC)
        auto* init_loop = dyn_cast<structured_control_flow::Map*>(&k_tile_body.at(0));
        EXPECT_NE(init_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), MC)));

        // Check nested second dimension
        auto& init_inner_body = init_loop->root();
        EXPECT_EQ(init_inner_body.size(), 1);
        auto* init_inner = dyn_cast<structured_control_flow::Map*>(&init_inner_body.at(0));
        EXPECT_NE(init_inner, nullptr);
        EXPECT_TRUE(symbolic::eq(init_inner->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(init_inner->condition(), symbolic::Lt(init_inner->indvar(), KC)));

        // Compute loop preserved
        auto* compute_loop = dyn_cast<structured_control_flow::For*>(&k_tile_body.at(1));
        EXPECT_NE(compute_loop, nullptr);

        // Writeback: nested for loops (MC x KC)
        auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&k_tile_body.at(2));
        EXPECT_NE(wb_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), MC)));

        // Verify the compute memlet uses LOCAL indices: (i-i_tile)*KC + (k-k_tile)
        auto& compute_i_body = compute_loop->root();
        EXPECT_GE(compute_i_body.size(), 1u);
        auto* compute_k_loop = dyn_cast<structured_control_flow::For*>(&compute_i_body.at(0));
        EXPECT_NE(compute_k_loop, nullptr);
        auto& compute_k_body = compute_k_loop->root();
        EXPECT_EQ(compute_k_body.size(), 1u);
        auto* compute_block = dyn_cast<structured_control_flow::Block*>(&compute_k_body.at(0));
        EXPECT_NE(compute_block, nullptr);

        bool found_local_read = false;
        bool found_local_write = false;
        for (auto* node : compute_block->dataflow().data_nodes()) {
            if (node->data() == "__daisy_out_local_storage_C0") {
                auto expected = symbolic::add(symbolic::mul(symbolic::sub(i, i_tile), KC), symbolic::sub(k, k_tile));
                // Check outgoing memlets (reads from this access node)
                for (auto& memlet : compute_block->dataflow().out_edges(*node)) {
                    found_local_read = true;
                    auto& subset = memlet.subset();
                    EXPECT_EQ(subset.size(), 1u);
                    EXPECT_TRUE(symbolic::eq(subset.at(0), expected));
                }
                // Check incoming memlets (writes to this access node)
                for (auto& memlet : compute_block->dataflow().in_edges(*node)) {
                    found_local_write = true;
                    auto& subset = memlet.subset();
                    EXPECT_EQ(subset.size(), 1u);
                    EXPECT_TRUE(symbolic::eq(subset.at(0), expected));
                }
            }
        }
        EXPECT_TRUE(found_local_read);
        EXPECT_TRUE(found_local_write);
    }
}

/**
 * Test: OutLocalStorage write-only with tiled loop (no init, only writeback)
 *
 * Before:
 *   for i_tile = 0..N step TILE:
 *       for i = i_tile..min(i_tile+TILE, N):
 *           C[i] = f(A[i])   (write-only to C)
 *
 * After OutLocalStorage(inner_loop, C):
 *   for i_tile = 0..N step TILE:
 *       for i = i_tile..min(i_tile+TILE, N):
 *           C_local[i - i_tile] = f(A[i])
 *       for d0 = 0..TILE:
 *           C[i_tile + d0] = C_local[d0]   // writeback only
 *
 * No init loop because C is write-only within inner_loop scope.
 */
TEST(OutLocalStorageTest, TiledWriteOnly_1D) {
    builder::StructuredSDFGBuilder builder("ols_tiled_wo", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("i", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto TILE = symbolic::integer(32);
    auto N = symbolic::symbol("N");
    auto i_tile = symbolic::symbol("i_tile");
    auto i = symbolic::symbol("i");

    // for i_tile = 0; i_tile < N; i_tile += TILE
    auto& tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, TILE));

    // for i = i_tile; i < min(i_tile+TILE, N); i++
    auto& inner_loop = builder.add_for(
        tile_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, TILE)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // C[i] = A[i]  (write-only to C)
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {i}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {i}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply OutLocalStorage at inner_loop level (tile has extent TILE)
    transformations::OutLocalStorage transformation(inner_loop, c_out);

    bool can_apply = transformation.can_be_applied(builder_opt, am);
    EXPECT_TRUE(can_apply);

    if (can_apply) {
        transformation.apply(builder_opt, am);


        // Verify local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

        // Structure: tile_loop body should be [inner_loop, writeback_loop] — NO init!
        auto& tile_body = tile_loop.root();
        EXPECT_EQ(tile_body.size(), 2);

        // First: compute loop
        auto* compute_loop = dyn_cast<structured_control_flow::For*>(&tile_body.at(0));
        EXPECT_NE(compute_loop, nullptr);

        // Second: writeback loop (0..TILE)
        auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&tile_body.at(1));
        EXPECT_NE(wb_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), TILE)));

        // Writeback body: C_local → assign → C
        auto& wb_body = wb_loop->root();
        EXPECT_EQ(wb_body.size(), 1);
        auto* wb_block = dyn_cast<structured_control_flow::Block*>(&wb_body.at(0));
        EXPECT_NE(wb_block, nullptr);
        bool has_c = false, has_local = false;
        for (auto* node : wb_block->dataflow().data_nodes()) {
            if (node->data() == "C") has_c = true;
            if (node->data() == "__daisy_out_local_storage_C0") has_local = true;
        }
        EXPECT_TRUE(has_c);
        EXPECT_TRUE(has_local);
    }
}

/**
 * Test: OutLocalStorage with flat pointer linearized 1D access (non-zero base)
 *
 * Before:
 *   for i_tile = 0..N step TILE:
 *       for j = 0..M:
 *           for i = i_tile..min(i_tile+TILE, N):
 *               C[i] += A[j*N+i]
 *
 * After OutLocalStorage(j_loop, C):
 *   The tile of C at j_loop level has extent TILE, base i_tile.
 *   for i_tile = 0..N step TILE:
 *       for d0 = 0..TILE: C_local[d0] = C[i_tile+d0]    // init
 *       for j = 0..M:
 *           for i = i_tile..min(i_tile+TILE, N):
 *               C_local[i-i_tile] += A[j*N+i]
 *       for d0 = 0..TILE: C[i_tile+d0] = C_local[d0]    // writeback
 */
TEST(OutLocalStorageTest, TiledAccumulator_1D_NonZeroBase) {
    builder::StructuredSDFGBuilder builder("ols_tiled_1d_base", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i_tile", sym_desc);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Double);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto TILE = symbolic::integer(64);
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto i_tile = symbolic::symbol("i_tile");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    // for i_tile = 0; i_tile < N; i_tile += TILE
    auto& tile_loop =
        builder.add_for(root, i_tile, symbolic::Lt(i_tile, N), symbolic::integer(0), symbolic::add(i_tile, TILE));

    // for j = 0; j < M; j++
    auto& j_loop =
        builder
            .add_for(tile_loop.root(), j, symbolic::Lt(j, M), symbolic::integer(0), symbolic::add(j, symbolic::one()));

    // for i = i_tile; i < min(i_tile+TILE, N); i++
    auto& i_loop = builder.add_for(
        j_loop.root(),
        i,
        symbolic::And(symbolic::Lt(i, symbolic::add(i_tile, TILE)), symbolic::Lt(i, N)),
        i_tile,
        symbolic::add(i, symbolic::one())
    );

    // C[i] += A[j*N+i]
    auto& block = builder.add_block(i_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {i}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::add(symbolic::mul(j, N), i)}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {i}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply OutLocalStorage at j_loop level (tile has extent TILE from inner i_loop)
    transformations::OutLocalStorage transformation(j_loop, c_in);

    bool can_apply = transformation.can_be_applied(builder_opt, am);
    EXPECT_TRUE(can_apply);

    if (can_apply) {
        transformation.apply(builder_opt, am);


        // Verify local buffer was created
        EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

        // Structure: tile_loop body should be [init_loop, j_loop, writeback_loop]
        auto& tile_body = tile_loop.root();
        EXPECT_EQ(tile_body.size(), 3);

        // Init loop: 0..TILE
        auto* init_loop = dyn_cast<structured_control_flow::Map*>(&tile_body.at(0));
        EXPECT_NE(init_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), TILE)));

        // Init body should read from C and write to C_local
        auto& init_body = init_loop->root();
        EXPECT_EQ(init_body.size(), 1);
        auto* init_block = dyn_cast<structured_control_flow::Block*>(&init_body.at(0));
        EXPECT_NE(init_block, nullptr);
        bool has_c = false, has_local = false;
        for (auto* node : init_block->dataflow().data_nodes()) {
            if (node->data() == "C") has_c = true;
            if (node->data() == "__daisy_out_local_storage_C0") has_local = true;
        }
        EXPECT_TRUE(has_c);
        EXPECT_TRUE(has_local);

        // Compute loop (j_loop) preserved
        auto* compute_loop = dyn_cast<structured_control_flow::For*>(&tile_body.at(1));
        EXPECT_NE(compute_loop, nullptr);

        // Writeback loop: 0..TILE
        auto* wb_loop = dyn_cast<structured_control_flow::Map*>(&tile_body.at(2));
        EXPECT_NE(wb_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), TILE)));
    }
}

// =========================================================================
// GPU Cooperative Path Tests
// =========================================================================

/**
 * Test: OutLocalStorage with NV_Shared rejects when no cooperative dimension exists
 *
 * Setup: GPU Map X (i, 0..N) → For k = 0..K, writing to C[i*K + k]
 * Tile bases depend on i (the only GPU dim), so no cooperative dim → rejected.
 */
TEST(OutLocalStorageTest, GPU_NoCoop_Rejected) {
    builder::StructuredSDFGBuilder builder("ols_gpu_nocoop", FunctionType_CPU);
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("k")}, ptr);
    // C[i*K + k] — base depends on i (GPU indvar)
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        c_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("i"), K), symbolic::symbol("k"))},
        ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage ols(loop, c_out, types::StorageType::NV_Shared());
    EXPECT_FALSE(ols.can_be_applied(builder_opt, am));
}

/**
 * Test: OutLocalStorage with NV_Shared rejects when applied to the outermost loop
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..16
 *   C[j*16 + k] is written — the nested structure is cooperative (i not in base),
 *   so the inner For would be a valid staging target.
 *
 * However, the transformation is applied to the OUTERMOST loop (Map X), which is
 * the CUDA kernel itself. Staging into shared memory at this level would place the
 * copy-out loops outside the kernel and force the per-block __shared__ buffer to be
 * passed across the kernel boundary as an argument — which is illegal in CUDA.
 * Hence it must be rejected.
 */
TEST(OutLocalStorageTest, GPU_OutermostLoop_Rejected) {
    builder::StructuredSDFGBuilder builder("ols_gpu_outermost", FunctionType_CPU);
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

    // GPU Map X: i = 0..N (block_size=32) — outermost loop / kernel boundary
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("k")}, ptr);
    // C[j*16 + k] — base depends on j only; i (the kernel/X dim) is free → cooperative
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        c_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("j"), symbolic::integer(16)), symbolic::symbol("k"))},
        ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Applied to the outermost loop (Map X = kernel) → must be rejected for NV_Shared
    transformations::OutLocalStorage ols(map_x, c_out, types::StorageType::NV_Shared());
    EXPECT_FALSE(ols.can_be_applied(builder_opt, am));
}

/**
 * Test: OutLocalStorage with NV_Shared on a flat pointer with cooperative writeback
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..M
 *   C[j*M + k] — linearized pointer access, writes depend on j + loop var k
 *
 * N, M symbolic. Tile bases = [j*M], extent = [M]. After substitution M→8.
 * X-dim (i) NOT in base → cooperative.
 */
TEST(OutLocalStorageTest, GPU_Cooperative_FlatPointer) {
    builder::StructuredSDFGBuilder builder("ols_gpu_coop", FunctionType_CPU);
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

    // For loop: k = 0..M (symbolic, same as Y-dim → resolves to 8)
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, ptr);
    // C[j*M + k] — base depends on j only; i is free → cooperative
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        c_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("j"), M), symbolic::symbol("k"))},
        ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage ols(loop, c_out, types::StorageType::NV_Shared());
    EXPECT_TRUE(ols.can_be_applied(builder_opt, am));
    ols.apply(builder_opt, am);


    // Verify: shared buffer was created with resolved size (M→8)
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));
    auto& buf_type = builder_opt.subject().type("__daisy_out_local_storage_C0");
    EXPECT_EQ(buf_type.storage_type(), types::StorageType::NV_Shared());
    auto& arr_type = static_cast<const types::Array&>(buf_type);
    // Per-thread Y dim (j in C base) contributes BY=8 slots; varying dim (M→8) contributes 8.
    // Total = 8 * 8 = 64.
    EXPECT_TRUE(symbolic::eq(arr_type.num_elements(), symbolic::integer(64)));

    // Verify structure: write-only → [main_loop, barrier, writeback_loop, barrier]
    auto& map_y_body = map_y.root();
    EXPECT_GE(map_y_body.size(), 4u);

    auto* main_loop = dyn_cast<structured_control_flow::For*>(&map_y_body.at(0));
    EXPECT_NE(main_loop, nullptr);

    auto* barrier1 = dyn_cast<structured_control_flow::Block*>(&map_y_body.at(1));
    EXPECT_NE(barrier1, nullptr);

    auto* wb_map = dyn_cast<structured_control_flow::Map*>(&map_y_body.at(2));
    EXPECT_NE(wb_map, nullptr);

    auto* barrier2 = dyn_cast<structured_control_flow::Block*>(&map_y_body.at(3));
    EXPECT_NE(barrier2, nullptr);
}

/**
 * Test: OutLocalStorage with NV_Shared read-write (has_read = true)
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..N
 *   C[j*N + k] is both read and written (accumulation: C[j*N+k] += A[i])
 *
 * N, M symbolic. Extent = N, resolved to 32 from X-dim. i NOT in base → cooperative.
 */
TEST(OutLocalStorageTest, GPU_Cooperative_ReadWrite) {
    builder::StructuredSDFGBuilder builder("ols_gpu_rw", FunctionType_CPU);
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

    // For loop: k = 0..N (symbolic, resolves to 32)
    auto& loop = builder.add_for(
        map_y.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), N),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {symbolic::symbol("i")}, ptr);
    // C[j*N + k] — both read and written (accumulation), symbolic stride N
    auto c_subset = symbolic::add(symbolic::mul(symbolic::symbol("j"), N), symbolic::symbol("k"));
    builder.add_computational_memlet(block, c_in, tasklet, "_in1", {c_subset}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {c_subset}, ptr);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage ols(loop, c_out, types::StorageType::NV_Shared());
    EXPECT_TRUE(ols.can_be_applied(builder_opt, am));
    ols.apply(builder_opt, am);


    // Verify: shared buffer created with resolved size (N→32)
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));
    auto& buf_type = builder_opt.subject().type("__daisy_out_local_storage_C0");
    EXPECT_EQ(buf_type.storage_type(), types::StorageType::NV_Shared());
    EXPECT_EQ(buf_type.type_id(), types::TypeID::Array);

    auto& arr_type = static_cast<const types::Array&>(buf_type);
    // Per-thread Y dim (j in C base) contributes BY=8 slots; varying dim (N→32) contributes 32.
    // Total = 8 * 32 = 256.
    EXPECT_TRUE(symbolic::eq(arr_type.num_elements(), symbolic::integer(256)));

    // Verify structure: has_read → [barrier, init_copy, barrier, main_loop, barrier, writeback, barrier]
    auto& map_y_body = map_y.root();
    EXPECT_GE(map_y_body.size(), 7u);

    auto* b1 = dyn_cast<structured_control_flow::Block*>(&map_y_body.at(0));
    EXPECT_NE(b1, nullptr);
    auto* init_map = dyn_cast<structured_control_flow::Map*>(&map_y_body.at(1));
    EXPECT_NE(init_map, nullptr);
    auto* b2 = dyn_cast<structured_control_flow::Block*>(&map_y_body.at(2));
    EXPECT_NE(b2, nullptr);
    auto* main_loop = dyn_cast<structured_control_flow::For*>(&map_y_body.at(3));
    EXPECT_NE(main_loop, nullptr);
    auto* b3 = dyn_cast<structured_control_flow::Block*>(&map_y_body.at(4));
    EXPECT_NE(b3, nullptr);
    auto* wb_map = dyn_cast<structured_control_flow::Map*>(&map_y_body.at(5));
    EXPECT_NE(wb_map, nullptr);
    auto* b4 = dyn_cast<structured_control_flow::Block*>(&map_y_body.at(6));
    EXPECT_NE(b4, nullptr);
}

/**
 * Test: OutLocalStorage with NV_Shared and both GPU dims cooperative
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..N
 *   C[k] — access does not depend on any GPU dim
 *
 * Neither i nor j appear in bases → both cooperative. Extent N resolves to 32.
 */
TEST(OutLocalStorageTest, GPU_Cooperative_AllDimsFree) {
    builder::StructuredSDFGBuilder builder("ols_gpu_allfree", FunctionType_CPU);
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

    // For loop: k = 0..N (resolves to 32)
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
    builder.add_computational_memlet(
        block, a_in, tasklet, "_in", {symbolic::add(symbolic::mul(symbolic::symbol("i"), M), symbolic::symbol("j"))}, ptr
    );
    // C[k] — no GPU indvar in write access
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("k")}, ptr);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage ols(loop, c_out, types::StorageType::NV_Shared());
    EXPECT_TRUE(ols.can_be_applied(builder_opt, am));
    ols.apply(builder_opt, am);


    // Verify buffer: extent N resolved to 32
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));
    auto& buf_type = builder_opt.subject().type("__daisy_out_local_storage_C0");
    auto& arr_type = static_cast<const types::Array&>(buf_type);
    EXPECT_TRUE(symbolic::eq(arr_type.num_elements(), symbolic::integer(32)));
    EXPECT_EQ(buf_type.storage_type(), types::StorageType::NV_Shared());
}

/**
 * Test: OutLocalStorage with NV_Shared rejects when extent is unresolvable symbolic
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → For k = 0..K
 *   C[k] — extent K is NOT a bound of any GPU map → stays symbolic → rejected
 */
TEST(OutLocalStorageTest, GPU_SymbolicExtent_Unresolvable_Rejected) {
    builder::StructuredSDFGBuilder builder("ols_gpu_unresolvable", FunctionType_CPU);
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

    // For loop: k = 0..K (K not a GPU bound → unresolvable)
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, ptr);
    // C[k] — base is 0, does not depend on i → cooperative, but extent K is unresolvable
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("k")}, ptr);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage ols(loop, c_out, types::StorageType::NV_Shared());
    EXPECT_FALSE(ols.can_be_applied(builder_opt, am));
}

/**
 * Test: OutLocalStorage with NV_Shared resolves symbolic extent from Y-dim
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..M
 *   C[i*M + k] — base depends on i (X), extent M resolves from Y-dim to 8
 *
 * X-dim i is in base → not cooperative on X. But j not in base → cooperative on Y.
 */
TEST(OutLocalStorageTest, GPU_Cooperative_SymbolicBounds) {
    builder::StructuredSDFGBuilder builder("ols_gpu_symbolic", FunctionType_CPU);
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

    // For loop: k = 0..M (symbolic, resolves to 8 from Y-dim)
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("j")}, ptr);
    // C[i*M + k] — base = i*M, depends on i (X-dim), extent = M
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        c_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("i"), M), symbolic::symbol("k"))},
        ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage ols(loop, c_out, types::StorageType::NV_Shared());
    EXPECT_TRUE(ols.can_be_applied(builder_opt, am));
    ols.apply(builder_opt, am);


    // Verify: buffer created with M→8
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));
    auto& buf_type = builder_opt.subject().type("__daisy_out_local_storage_C0");
    auto& arr_type = static_cast<const types::Array&>(buf_type);
    // Per-thread X dim (i in C base) contributes BX=32 slots; varying dim (M→8) contributes 8.
    // Total = 32 * 8 = 256.
    EXPECT_TRUE(symbolic::eq(arr_type.num_elements(), symbolic::integer(256)));
    EXPECT_EQ(buf_type.storage_type(), types::StorageType::NV_Shared());
}

/**
 * Test: OutLocalStorage with CPU_Stack (default) rejects when inside a GPU region.
 *
 * Setup:
 *   Map X (i, 0..N, CUDA block_size=32) → For k = 0..K
 *   C[i*K + k] = A[k]
 *
 * Applying OutLocalStorage with CPU_Stack on the For loop inside the GPU Map
 * should fail because CPU_Stack is invalid inside a GPU kernel (the codegen
 * would emit a stack array in device code, which is not a valid device pointer).
 */
TEST(OutLocalStorageTest, GPU_CPUStack_InsideGPU_Rejected) {
    builder::StructuredSDFGBuilder builder("ols_cpustack_in_gpu", FunctionType_CPU);
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
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("k")}, ptr);
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        c_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("i"), K), symbolic::symbol("k"))},
        ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // CPU_Stack (default) inside GPU region → must be rejected
    transformations::OutLocalStorage ols(loop, c_out);
    EXPECT_FALSE(ols.can_be_applied(builder_opt, am));
}

/**
 * Test: OutLocalStorage with CPU_Stack rejects when applied to the outermost
 * GPU-scheduled map itself.
 *
 * Setup:
 *   Map X (i, 0..32, CUDA) → For k = 0..4
 *   C[i*4 + k] = A[k]
 *
 * The GPU indvar 'i' appears in C's tile bases (per-thread), so there is no
 * cooperative dimension — the existing cooperative check would NOT reject this.
 * However, applying CPU_Stack OLS to the outermost CUDA map would place the
 * init/writeback copies outside the kernel on the host, which is invalid.
 */
TEST(OutLocalStorageTest, GPU_CPUStack_OutermostCUDAMap_Rejected) {
    builder::StructuredSDFGBuilder builder("ols_cpustack_outermost_cuda", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    // GPU Map X: i = 0..32 (outermost — this is the kernel boundary)
    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_x = builder.add_map(
        seq,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        sched_x
    );

    // For loop k = 0..4
    auto& loop = builder.add_for(
        map_x.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("k")}, ptr);
    // C[i*4 + k] — base depends on 'i' (GPU indvar), so per-thread (no coop dim)
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        c_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::integer(4)), symbolic::symbol("k"))},
        ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // CPU_Stack applied to the outermost CUDA map → must be rejected
    transformations::OutLocalStorage ols(map_x, c_out);
    EXPECT_FALSE(ols.can_be_applied(builder_opt, am));
}

// CUDA map wrapped by a regular For loop — still the kernel boundary.
// CPU_Stack must be rejected because the buffer would be host-allocated.
TEST(OutLocalStorageTest, GPU_CPUStack_CUDAMapWrappedByFor_Rejected) {
    builder::StructuredSDFGBuilder builder("ols_cpustack_cuda_wrapped_for", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("n", loop_var);

    // Outer For loop n = 0..2 (regular, not GPU)
    auto& outer_for = builder.add_for(
        seq,
        symbolic::symbol("n"),
        symbolic::Lt(symbolic::symbol("n"), symbolic::integer(2)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("n"), symbolic::integer(1))
    );

    // GPU Map: i = 0..32 (kernel boundary — no GPU ancestors)
    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_x = builder.add_map(
        outer_for.root(),
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        sched_x
    );

    // For loop k = 0..4
    auto& loop = builder.add_for(
        map_x.root(),
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    auto& block = builder.add_block(loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("k")}, ptr);
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        c_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::integer(4)), symbolic::symbol("k"))},
        ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // CPU_Stack on CUDA map wrapped by For — still kernel boundary, must reject
    transformations::OutLocalStorage ols(map_x, c_out);
    EXPECT_FALSE(ols.can_be_applied(builder_opt, am));
}

// For loop wrapping a CUDA map — CPU_Stack applied to the For loop itself.
// Buffer would be host-allocated but referenced inside the descendant kernel.
TEST(OutLocalStorageTest, GPU_CPUStack_ForContainingCUDAMap_Rejected) {
    builder::StructuredSDFGBuilder builder("ols_cpustack_for_contains_cuda", FunctionType_CPU);
    auto& seq = builder.subject().root();

    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);

    builder.add_container("A", ptr, true);
    builder.add_container("C", ptr);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    // Outer For loop k = 0..4 (regular, not GPU)
    auto& outer_for = builder.add_for(
        seq,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::integer(4)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1))
    );

    // GPU Map: i = 0..32 (inside the For loop)
    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_x = builder.add_map(
        outer_for.root(),
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        sched_x
    );

    // C[i*4 + k] write inside GPU map
    auto& block = builder.add_block(map_x.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("k")}, ptr);
    builder.add_computational_memlet(
        block,
        tasklet,
        "_out",
        c_out,
        {symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::integer(4)), symbolic::symbol("k"))},
        ptr
    );

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // CPU_Stack on the For loop that contains a CUDA map — must reject
    transformations::OutLocalStorage ols(outer_for, c_out);
    EXPECT_FALSE(ols.can_be_applied(builder_opt, am));
}
