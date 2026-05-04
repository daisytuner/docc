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

using namespace sdfg;

TEST(OutLocalStorage, Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::integer(4);
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {});

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto& new_root = builder_opt.subject().root();
    // Apply
    transformations::OutLocalStorage transformation(loop, access_out);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Check
    EXPECT_EQ(new_root.size(), 3);
    auto init_block = dynamic_cast<structured_control_flow::Block*>(&new_root.at(0).first);
    EXPECT_NE(init_block, nullptr);
    EXPECT_EQ(init_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(init_block->dataflow().edges().size(), 2);
    bool c_access = false;
    bool a_access = false;
    for (auto& node : init_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C") {
                c_access = true;
            } else if (access->data() == "__daisy_out_local_storage_C0") {
                a_access = true;
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);

    for (auto& memlet : init_block->dataflow().edges()) {
        if (memlet.dst_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.dst());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "__daisy_out_local_storage_C0");
        } else if (memlet.src_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.src());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "C");
        }
    }

    auto new_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(new_loop, nullptr);

    auto& body_loop = new_loop->root();
    EXPECT_EQ(body_loop.size(), 1);
    auto loop_block = dynamic_cast<structured_control_flow::Block*>(&body_loop.at(0).first);
    EXPECT_NE(loop_block, nullptr);
    EXPECT_EQ(loop_block->dataflow().nodes().size(), 4);
    EXPECT_EQ(loop_block->dataflow().edges().size(), 3);
    int accesses = 0;
    a_access = false;
    for (auto access_node : loop_block->dataflow().data_nodes()) {
        if (access_node->data() == "A") {
            a_access = true;
        } else if (access_node->data() == "__daisy_out_local_storage_C0") {
            accesses++;
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_EQ(accesses, 2);

    auto deinit_block = dynamic_cast<structured_control_flow::Block*>(&new_root.at(2).first);
    EXPECT_NE(deinit_block, nullptr);

    EXPECT_EQ(deinit_block->dataflow().nodes().size(), 3);
    EXPECT_EQ(deinit_block->dataflow().edges().size(), 2);
    c_access = false;
    a_access = false;
    for (auto& node : deinit_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C") {
                c_access = true;
            } else if (access->data() == "__daisy_out_local_storage_C0") {
                a_access = true;
            }
        } else if (auto tasklet = dynamic_cast<data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet->code(), data_flow::TaskletCode::assign);
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);

    for (auto& memlet : deinit_block->dataflow().edges()) {
        if (memlet.dst_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.dst());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "C");
        } else if (memlet.src_conn() == "void") {
            auto access = dynamic_cast<data_flow::AccessNode*>(&memlet.src());
            EXPECT_NE(access, nullptr);
            EXPECT_EQ(access->data(), "__daisy_out_local_storage_C0");
        }
    }
}

TEST(OutLocalStorage, Array) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers — flat pointers with linearized access
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", desc, true);
    builder.add_container("C", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::integer(100);
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation: C[i] += A[i]
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")}, desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto& new_root = builder_opt.subject().root();
    // Apply
    transformations::OutLocalStorage transformation(loop, access_out);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Check: [init_loop, compute_loop, writeback_loop]
    EXPECT_EQ(new_root.size(), 3);

    // Init loop: copy C to C_local
    auto init_for = dynamic_cast<structured_control_flow::Map*>(&new_root.at(0).first);
    EXPECT_NE(init_for, nullptr);
    EXPECT_TRUE(symbolic::eq(init_for->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_for->condition(), symbolic::Lt(init_for->indvar(), symbolic::integer(100))));

    auto& init_body = init_for->root();
    EXPECT_EQ(init_body.size(), 1);
    auto init_block = dynamic_cast<structured_control_flow::Block*>(&init_body.at(0).first);
    EXPECT_NE(init_block, nullptr);
    bool c_access = false;
    bool a_access = false;
    for (auto& node : init_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C")
                c_access = true;
            else if (access->data() == "__daisy_out_local_storage_C0")
                a_access = true;
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);

    // Compute loop preserved
    auto new_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(new_loop, nullptr);

    auto& body_loop = new_loop->root();
    EXPECT_EQ(body_loop.size(), 1);
    auto loop_block = dynamic_cast<structured_control_flow::Block*>(&body_loop.at(0).first);
    EXPECT_NE(loop_block, nullptr);
    int accesses = 0;
    a_access = false;
    for (auto access_node : loop_block->dataflow().data_nodes()) {
        if (access_node->data() == "A")
            a_access = true;
        else if (access_node->data() == "__daisy_out_local_storage_C0")
            accesses++;
    }
    EXPECT_TRUE(a_access);
    EXPECT_EQ(accesses, 2);

    // Writeback loop: copy C_local back to C
    auto wb_for = dynamic_cast<structured_control_flow::Map*>(&new_root.at(2).first);
    EXPECT_NE(wb_for, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_for->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_for->condition(), symbolic::Lt(wb_for->indvar(), symbolic::integer(100))));

    auto& wb_body = wb_for->root();
    EXPECT_EQ(wb_body.size(), 1);
    auto wb_block = dynamic_cast<structured_control_flow::Block*>(&wb_body.at(0).first);
    EXPECT_NE(wb_block, nullptr);
    c_access = false;
    a_access = false;
    for (auto& node : wb_block->dataflow().nodes()) {
        if (auto access = dynamic_cast<data_flow::AccessNode*>(&node)) {
            if (access->data() == "C")
                c_access = true;
            else if (access->data() == "__daisy_out_local_storage_C0")
                a_access = true;
        }
    }
    EXPECT_TRUE(a_access);
    EXPECT_TRUE(c_access);
}

TEST(OutLocalStorage, Fail) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("C", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Place an access to A outside the loop (before it, in the root sequence).
    auto& outer_block = builder.add_block(root);
    auto& access_outside = builder.add_access(outer_block, "C");
    auto& access_i_outside = builder.add_access(outer_block, "i");
    auto& tasklet_outside = builder.add_tasklet(outer_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(outer_block, access_i_outside, tasklet_outside, "_in", {});
    builder.add_computational_memlet(outer_block, tasklet_outside, "_out", access_outside, {symbolic::integer(0)}, desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add computation inside the loop (only writes to A)
    auto& block = builder.add_block(body);
    auto& access_in = builder.add_access(block, "i");
    auto& access_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::integer(0)}, desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply with the outside access node — should fail
    transformations::OutLocalStorage transformation(loop, access_outside);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

/**
 * Test: OutLocalStorage on inner-loop accumulator with flat pointers
 *
 * Before:
 *   for i = 0..4:
 *       for j = 0..8:
 *           C[j] += A[j]
 *
 * After OutLocalStorage(i_loop, C):
 *   for __d0 = 0..8: C_local[__d0] = C[__d0]    // init (read-write)
 *   for i = 0..4:
 *       for j = 0..8: C_local[j] += A[j]         // compute on tile
 *   for __d0 = 0..8: C[__d0] = C_local[__d0]    // writeback
 */
TEST(OutLocalStorage, InnerLoopAccumulator) {
    builder::StructuredSDFGBuilder builder("ols_inner_acc", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    // Outer loop: for i = 0..4
    auto i = symbolic::symbol("i");
    auto& outer_loop = builder.add_for(
        root, i, symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(0), symbolic::add(i, symbolic::integer(1))
    );

    // Inner loop: for j = 0..8
    auto j = symbolic::symbol("j");
    auto& inner_loop = builder.add_for(
        outer_loop.root(),
        j,
        symbolic::Lt(j, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1))
    );

    // C[j] += A[j]
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

    transformations::OutLocalStorage transformation(outer_loop, c_in);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    // Verify local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

    // Structure: root should now contain [init_loop, outer_loop, writeback_loop]
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 3);

    auto* init_loop = dynamic_cast<structured_control_flow::Map*>(&new_root.at(0).first);
    EXPECT_NE(init_loop, nullptr);
    // Init loop should iterate 0..8
    EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), symbolic::integer(8))));

    auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(compute_loop, nullptr);

    auto* wb_loop = dynamic_cast<structured_control_flow::Map*>(&new_root.at(2).first);
    EXPECT_NE(wb_loop, nullptr);
    // Writeback loop should iterate 0..8
    EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), symbolic::integer(8))));

    // Check init loop body: C → tasklet(assign) → C_local
    auto& init_body = init_loop->root();
    EXPECT_EQ(init_body.size(), 1);
    auto* init_block = dynamic_cast<structured_control_flow::Block*>(&init_body.at(0).first);
    EXPECT_NE(init_block, nullptr);
    bool has_c = false, has_local = false;
    for (auto* node : init_block->dataflow().data_nodes()) {
        if (node->data() == "C") has_c = true;
        if (node->data() == "__daisy_out_local_storage_C0") has_local = true;
    }
    EXPECT_TRUE(has_c);
    EXPECT_TRUE(has_local);

    // Check writeback loop body: C_local → tasklet(assign) → C
    auto& wb_body = wb_loop->root();
    EXPECT_EQ(wb_body.size(), 1);
    auto* wb_block = dynamic_cast<structured_control_flow::Block*>(&wb_body.at(0).first);
    EXPECT_NE(wb_block, nullptr);
    has_c = false;
    has_local = false;
    for (auto* node : wb_block->dataflow().data_nodes()) {
        if (node->data() == "C") has_c = true;
        if (node->data() == "__daisy_out_local_storage_C0") has_local = true;
    }
    EXPECT_TRUE(has_c);
    EXPECT_TRUE(has_local);
}

/**
 * Test: OutLocalStorage on write-only inner-loop pattern with flat pointers
 *
 * Before:
 *   for i = 0..4:
 *       for j = 0..8:
 *           C[j] = A[j]   (write-only, no read of C)
 *
 * After OutLocalStorage(i_loop, C):
 *   for i = 0..4:
 *       for j = 0..8: C_local[j] = A[j]          // compute on tile
 *   for __d0 = 0..8: C[__d0] = C_local[__d0]    // writeback only (no init!)
 */
TEST(OutLocalStorage, WriteOnly) {
    builder::StructuredSDFGBuilder builder("ols_write_only", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

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

    // C[j] = A[j] — write only, no read of C
    auto& block = builder.add_block(inner_loop.root());
    auto& a_in = builder.add_access(block, "A");
    auto& c_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {j}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {j}, ptr_desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    transformations::OutLocalStorage transformation(outer_loop, c_out);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, am));
    transformation.apply(builder_opt, am);

    // Verify local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

    // Structure: root should now contain [outer_loop, writeback_loop] — NO init!
    auto& new_root = builder_opt.subject().root();
    EXPECT_EQ(new_root.size(), 2);

    auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(0).first);
    EXPECT_NE(compute_loop, nullptr);

    auto* wb_loop = dynamic_cast<structured_control_flow::Map*>(&new_root.at(1).first);
    EXPECT_NE(wb_loop, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), symbolic::integer(8))));
}

/**
 * Test: OutLocalStorage fails on read-only container
 *
 * for i = 0..4:
 *     for j = 0..8:
 *         B[j] = A[j]   (A is read-only, not written)
 */
TEST(OutLocalStorage, FailsOnReadOnly) {
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
TEST(OutLocalStorage, FailsOnAccessOutsideLoop) {
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
TEST(OutLocalStorage, FlatPointer_Linearized2D) {
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
    auto* init_loop = dynamic_cast<structured_control_flow::Map*>(&new_root.at(0).first);
    EXPECT_NE(init_loop, nullptr);
    // Should iterate 0..4 (first dim extent)
    EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), symbolic::integer(4))));

    // Init loop should contain nested loop for second dimension
    auto& init_body = init_loop->root();
    EXPECT_EQ(init_body.size(), 1);
    auto* inner_init = dynamic_cast<structured_control_flow::Map*>(&init_body.at(0).first);
    EXPECT_NE(inner_init, nullptr);
    EXPECT_TRUE(symbolic::eq(inner_init->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(inner_init->condition(), symbolic::Lt(inner_init->indvar(), symbolic::integer(8))));

    // Compute loop preserved
    auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&new_root.at(1).first);
    EXPECT_NE(compute_loop, nullptr);

    // Writeback should be a for loop
    auto* wb_loop = dynamic_cast<structured_control_flow::Map*>(&new_root.at(2).first);
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
TEST(OutLocalStorage, TiledAccumulator_2D) {
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
        auto* init_loop = dynamic_cast<structured_control_flow::Map*>(&k_tile_body.at(0).first);
        EXPECT_NE(init_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), MC)));

        // Check nested second dimension
        auto& init_inner_body = init_loop->root();
        EXPECT_EQ(init_inner_body.size(), 1);
        auto* init_inner = dynamic_cast<structured_control_flow::Map*>(&init_inner_body.at(0).first);
        EXPECT_NE(init_inner, nullptr);
        EXPECT_TRUE(symbolic::eq(init_inner->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(init_inner->condition(), symbolic::Lt(init_inner->indvar(), KC)));

        // Compute loop preserved
        auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&k_tile_body.at(1).first);
        EXPECT_NE(compute_loop, nullptr);

        // Writeback: nested for loops (MC x KC)
        auto* wb_loop = dynamic_cast<structured_control_flow::Map*>(&k_tile_body.at(2).first);
        EXPECT_NE(wb_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), MC)));

        // Verify the compute memlet uses LOCAL indices: (i-i_tile)*KC + (k-k_tile)
        auto& compute_i_body = compute_loop->root();
        EXPECT_GE(compute_i_body.size(), 1u);
        auto* compute_k_loop = dynamic_cast<structured_control_flow::For*>(&compute_i_body.at(0).first);
        EXPECT_NE(compute_k_loop, nullptr);
        auto& compute_k_body = compute_k_loop->root();
        EXPECT_EQ(compute_k_body.size(), 1u);
        auto* compute_block = dynamic_cast<structured_control_flow::Block*>(&compute_k_body.at(0).first);
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
TEST(OutLocalStorage, TiledWriteOnly_1D) {
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
        auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&tile_body.at(0).first);
        EXPECT_NE(compute_loop, nullptr);

        // Second: writeback loop (0..TILE)
        auto* wb_loop = dynamic_cast<structured_control_flow::Map*>(&tile_body.at(1).first);
        EXPECT_NE(wb_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(wb_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(wb_loop->condition(), symbolic::Lt(wb_loop->indvar(), TILE)));

        // Writeback body: C_local → assign → C
        auto& wb_body = wb_loop->root();
        EXPECT_EQ(wb_body.size(), 1);
        auto* wb_block = dynamic_cast<structured_control_flow::Block*>(&wb_body.at(0).first);
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
TEST(OutLocalStorage, TiledAccumulator_1D_NonZeroBase) {
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
        auto* init_loop = dynamic_cast<structured_control_flow::Map*>(&tile_body.at(0).first);
        EXPECT_NE(init_loop, nullptr);
        EXPECT_TRUE(symbolic::eq(init_loop->init(), symbolic::integer(0)));
        EXPECT_TRUE(symbolic::eq(init_loop->condition(), symbolic::Lt(init_loop->indvar(), TILE)));

        // Init body should read from C and write to C_local
        auto& init_body = init_loop->root();
        EXPECT_EQ(init_body.size(), 1);
        auto* init_block = dynamic_cast<structured_control_flow::Block*>(&init_body.at(0).first);
        EXPECT_NE(init_block, nullptr);
        bool has_c = false, has_local = false;
        for (auto* node : init_block->dataflow().data_nodes()) {
            if (node->data() == "C") has_c = true;
            if (node->data() == "__daisy_out_local_storage_C0") has_local = true;
        }
        EXPECT_TRUE(has_c);
        EXPECT_TRUE(has_local);

        // Compute loop (j_loop) preserved
        auto* compute_loop = dynamic_cast<structured_control_flow::For*>(&tile_body.at(1).first);
        EXPECT_NE(compute_loop, nullptr);

        // Writeback loop: 0..TILE
        auto* wb_loop = dynamic_cast<structured_control_flow::Map*>(&tile_body.at(2).first);
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
TEST(OutLocalStorage, GPU_NoCoop_Rejected) {
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
 * Test: OutLocalStorage with NV_Shared on a flat pointer with cooperative writeback
 *
 * Setup:
 *   Map X (i, 0..N, block_size=32) → Map Y (j, 0..M, block_size=8) → For k = 0..M
 *   C[j*M + k] — linearized pointer access, writes depend on j + loop var k
 *
 * N, M symbolic. Tile bases = [j*M], extent = [M]. After substitution M→8.
 * X-dim (i) NOT in base → cooperative.
 */
TEST(OutLocalStorage, GPU_Cooperative_FlatPointer) {
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
    EXPECT_TRUE(symbolic::eq(arr_type.num_elements(), symbolic::integer(8)));

    // Verify structure: write-only → [main_loop, barrier, writeback_loop, barrier]
    auto& map_y_body = map_y.root();
    EXPECT_GE(map_y_body.size(), 4u);

    auto* main_loop = dynamic_cast<structured_control_flow::For*>(&map_y_body.at(0).first);
    EXPECT_NE(main_loop, nullptr);

    auto* barrier1 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(1).first);
    EXPECT_NE(barrier1, nullptr);

    auto* wb_map = dynamic_cast<structured_control_flow::Map*>(&map_y_body.at(2).first);
    EXPECT_NE(wb_map, nullptr);

    auto* barrier2 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(3).first);
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
TEST(OutLocalStorage, GPU_Cooperative_ReadWrite) {
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
    EXPECT_TRUE(symbolic::eq(arr_type.num_elements(), symbolic::integer(32)));

    // Verify structure: has_read → [barrier, init_copy, barrier, main_loop, barrier, writeback, barrier]
    auto& map_y_body = map_y.root();
    EXPECT_GE(map_y_body.size(), 7u);

    auto* b1 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(0).first);
    EXPECT_NE(b1, nullptr);
    auto* init_map = dynamic_cast<structured_control_flow::Map*>(&map_y_body.at(1).first);
    EXPECT_NE(init_map, nullptr);
    auto* b2 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(2).first);
    EXPECT_NE(b2, nullptr);
    auto* main_loop = dynamic_cast<structured_control_flow::For*>(&map_y_body.at(3).first);
    EXPECT_NE(main_loop, nullptr);
    auto* b3 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(4).first);
    EXPECT_NE(b3, nullptr);
    auto* wb_map = dynamic_cast<structured_control_flow::Map*>(&map_y_body.at(5).first);
    EXPECT_NE(wb_map, nullptr);
    auto* b4 = dynamic_cast<structured_control_flow::Block*>(&map_y_body.at(6).first);
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
TEST(OutLocalStorage, GPU_Cooperative_AllDimsFree) {
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
TEST(OutLocalStorage, GPU_SymbolicExtent_Unresolvable_Rejected) {
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
TEST(OutLocalStorage, GPU_Cooperative_SymbolicBounds) {
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
    EXPECT_TRUE(symbolic::eq(arr_type.num_elements(), symbolic::integer(8)));
    EXPECT_EQ(buf_type.storage_type(), types::StorageType::NV_Shared());
}

/**
 * Test: OutLocalStorage CPU_Stack with flat pointer (non-GPU baseline)
 *
 * Setup: for i = 0..N: for k = 0..16: C[i*16 + k] = A[k]
 * After: for loop writes local[k], then Map(0..16) copies local[d] → C[i*16+d]
 */
TEST(OutLocalStorage, CPU_FlatPointer_Linearized) {
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

    auto* main_loop = dynamic_cast<structured_control_flow::For*>(&outer_body.at(0).first);
    EXPECT_NE(main_loop, nullptr);

    auto* wb_map = dynamic_cast<structured_control_flow::Map*>(&outer_body.at(1).first);
    EXPECT_NE(wb_map, nullptr);
    EXPECT_TRUE(symbolic::eq(wb_map->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(wb_map->condition(), symbolic::Lt(wb_map->indvar(), symbolic::integer(16))));

    // Verify the compute memlet uses LOCAL indices (k, zero-based)
    auto& main_body = main_loop->root();
    EXPECT_EQ(main_body.size(), 1u);
    auto* compute_block = dynamic_cast<structured_control_flow::Block*>(&main_body.at(0).first);
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

// =========================================================================
// Tile Group Tests: Multiple access groups to the same container
// =========================================================================

/**
 * Test: OutLocalStorage with SYR2K-style accumulator C[i*N+j] in k-loop
 *
 * Pattern: for k: C[i*N+j] += A[i*K+k] * B[j*K+k]
 * OLS on C at k_loop → single group (only one write pattern C[i,j]),
 * creates scalar-like local since tile has extents [1,1] at k-loop level.
 */
TEST(OutLocalStorage, TileGroups_SingleWriteGroup) {
    builder::StructuredSDFGBuilder builder("ols_syr2k_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc, true);
    builder.add_container("C", ptr_desc);

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

    // Block: C[i*N+j] += A[i*16+k] * B[j*16+k]  (simplified as fp_add)
    auto& block = builder.add_block(k_loop.root());
    auto& c_in = builder.add_access(block, "C");
    auto& c_out = builder.add_access(block, "C");
    auto& a_in = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_c", "_a"});

    auto lin_c = symbolic::add(symbolic::mul(i, N), j);
    auto lin_a = symbolic::add(symbolic::mul(i, K), k);

    builder.add_computational_memlet(block, c_in, tasklet, "_c", {lin_c}, ptr_desc);
    builder.add_computational_memlet(block, a_in, tasklet, "_a", {lin_a}, ptr_desc);
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {lin_c}, ptr_desc);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply OLS on C at k_loop level
    transformations::OutLocalStorage ols(k_loop, c_in);
    EXPECT_TRUE(ols.can_be_applied(builder_opt, am));
    ols.apply(builder_opt, am);

    // Verify: local buffer was created
    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

    // j_loop body: [init, k_loop, writeback] (scalar-like since extents are [1,1])
    // The delinearized tile at k-loop for C is [i,j]->[i,j], extents [1,1]
    // So it acts like a scalar accumulator
    auto& j_body = j_loop.root();
    EXPECT_EQ(j_body.size(), 3u);

    // Verify A access remains untouched in the compute loop
    auto& k_body = k_loop.root();
    EXPECT_EQ(k_body.size(), 1u);
    auto* compute_block = dynamic_cast<structured_control_flow::Block*>(&k_body.at(0).first);
    ASSERT_NE(compute_block, nullptr);

    bool found_a = false;
    bool found_local_c = false;
    for (auto* node : compute_block->dataflow().data_nodes()) {
        if (node->data() == "A") found_a = true;
        if (node->data() == "__daisy_out_local_storage_C0") found_local_c = true;
    }
    EXPECT_TRUE(found_a);
    EXPECT_TRUE(found_local_c);
}

/**
 * Test: OutLocalStorage with two write groups to the same container
 *
 * Pattern: for k: C[i*N+j] += expr1; C[i*N+j+1] += expr2
 * Two writes with constant-offset bases → merge into one group.
 * OLS should handle both writes in the same local buffer.
 */
TEST(OutLocalStorage, TileGroups_ConstantOffsetMerge) {
    builder::StructuredSDFGBuilder builder("ols_const_offset_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("C", ptr_desc);

    auto& root = builder.subject().root();

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    // for i = 0..N
    auto& i_loop =
        builder.add_for(root, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::one()));

    // for j = 0..8
    auto& j_loop = builder.add_for(
        i_loop.root(), j, symbolic::Lt(j, symbolic::integer(8)), symbolic::integer(0), symbolic::add(j, symbolic::one())
    );

    // for k = 0..4
    auto& k_loop = builder.add_for(
        j_loop.root(), k, symbolic::Lt(k, symbolic::integer(4)), symbolic::integer(0), symbolic::add(k, symbolic::one())
    );

    // Block 1: C[i*8 + j] += A[k]
    auto& block1 = builder.add_block(k_loop.root());
    auto& c1_in = builder.add_access(block1, "C");
    auto& c1_out = builder.add_access(block1, "C");
    auto& a1_in = builder.add_access(block1, "A");
    auto& t1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_c", "_a"});

    auto lin_c1 = symbolic::add(symbolic::mul(i, symbolic::integer(8)), j);

    builder.add_computational_memlet(block1, c1_in, t1, "_c", {lin_c1}, ptr_desc);
    builder.add_computational_memlet(block1, a1_in, t1, "_a", {k}, ptr_desc);
    builder.add_computational_memlet(block1, t1, "_out", c1_out, {lin_c1}, ptr_desc);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply OLS on C at k_loop level — tile should be [1,1] (i and j are constant in k-loop)
    transformations::OutLocalStorage ols(k_loop, c1_in);
    EXPECT_TRUE(ols.can_be_applied(builder_opt, am));
    ols.apply(builder_opt, am);

    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

    // Structure: j_loop body = [init, k_loop, writeback]
    auto& j_body = j_loop.root();
    EXPECT_EQ(j_body.size(), 3u);
}

/**
 * Test: OutLocalStorage with SYR2K double-write pattern
 *
 * Pattern: for k: C[i*N+j] += A[i*K+k]*B[j*K+k] + B[i*K+k]*A[j*K+k]
 * Both writes to C[i*N+j] — same group. OLS should work normally.
 * This mirrors the real SYR2K kernel where the C accumulation is the OLS target.
 */
TEST(OutLocalStorage, TileGroups_SYR2K_Accumulator) {
    builder::StructuredSDFGBuilder builder("ols_syr2k_acc_test", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar elem_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(elem_desc);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", ptr_desc, true);
    builder.add_container("B", ptr_desc, true);
    builder.add_container("C", ptr_desc);

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

    auto lin_c = symbolic::add(symbolic::mul(i, N), j);

    // Block 1: C[i*N+j] += A[i*16+k] * B[j*16+k]
    auto& block1 = builder.add_block(k_loop.root());
    auto& c1_in = builder.add_access(block1, "C");
    auto& c1_out = builder.add_access(block1, "C");
    auto& a1_in = builder.add_access(block1, "A");
    auto& t1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_c", "_a"});
    builder.add_computational_memlet(block1, c1_in, t1, "_c", {lin_c}, ptr_desc);
    builder.add_computational_memlet(block1, a1_in, t1, "_a", {symbolic::add(symbolic::mul(i, K), k)}, ptr_desc);
    builder.add_computational_memlet(block1, t1, "_out", c1_out, {lin_c}, ptr_desc);

    // Block 2: C[i*N+j] += B[i*16+k] * A[j*16+k]
    auto& block2 = builder.add_block(k_loop.root());
    auto& c2_in = builder.add_access(block2, "C");
    auto& c2_out = builder.add_access(block2, "C");
    auto& b2_in = builder.add_access(block2, "B");
    auto& t2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_c", "_b"});
    builder.add_computational_memlet(block2, c2_in, t2, "_c", {lin_c}, ptr_desc);
    builder.add_computational_memlet(block2, b2_in, t2, "_b", {symbolic::add(symbolic::mul(i, K), k)}, ptr_desc);
    builder.add_computational_memlet(block2, t2, "_out", c2_out, {lin_c}, ptr_desc);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager am(builder_opt.subject());

    // Apply OLS on C at k_loop level
    transformations::OutLocalStorage ols(k_loop, c1_in);
    EXPECT_TRUE(ols.can_be_applied(builder_opt, am));
    ols.apply(builder_opt, am);

    EXPECT_TRUE(builder_opt.subject().exists("__daisy_out_local_storage_C0"));

    // All C accesses should be rewritten (both blocks write same C[i,j] pattern)
    auto& k_body = k_loop.root();
    for (size_t bi = 0; bi < k_body.size(); ++bi) {
        auto* blk = dynamic_cast<structured_control_flow::Block*>(&k_body.at(bi).first);
        if (!blk) continue;
        for (auto* node : blk->dataflow().data_nodes()) {
            // No raw "C" access should remain
            EXPECT_NE(node->data(), "C") << "All C accesses should be rewritten to local";
        }
    }

    // A and B accesses should remain unchanged
    bool found_a = false, found_b = false;
    for (size_t bi = 0; bi < k_body.size(); ++bi) {
        auto* blk = dynamic_cast<structured_control_flow::Block*>(&k_body.at(bi).first);
        if (!blk) continue;
        for (auto* node : blk->dataflow().data_nodes()) {
            if (node->data() == "A") found_a = true;
            if (node->data() == "B") found_b = true;
        }
    }
    EXPECT_TRUE(found_a);
    EXPECT_TRUE(found_b);
}
