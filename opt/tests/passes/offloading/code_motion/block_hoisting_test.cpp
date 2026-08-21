#include "sdfg/passes/offloading/code_motion/block_hoisting.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/alloca.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"
#include "sdfg/data_flow/library_nodes/stdlib/memmove.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(BlockHoistingTest, Map_InvariantMove) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant move
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder
            .add_dereference_memlet(block1, a, a_, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, Map_InvariantView) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant view
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder.add_reference_memlet(block1, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, Map_InvariantAlloca) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant alloca
    auto& block1 = builder.add_block(body);
    {
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<stdlib::AllocaNode>(block1, DebugInfo(), symbolic::one());
        builder.add_computational_memlet(block1, libnode, "_ret", c, {}, desc_ptr);
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::zero()});
    }
    auto& block3 = builder.add_block(body);
    {
        auto& B = builder.add_access(block3, "B");
        auto& A = builder.add_access(block3, "A");
        auto& tasklet1 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block3, B, tasklet1, "_in", {symbolic::symbol("i")});
        builder.add_computational_memlet(block3, tasklet1, "_out", A, {symbolic::symbol("i")});
    }
    auto& block4 = builder.add_block(body);
    {
        auto& c = builder.add_access(block4, "c");
        auto& B = builder.add_access(block4, "B");
        auto& tasklet1 = builder.add_tasklet(block4, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block4, c, tasklet1, "_in", {symbolic::zero()});
        builder.add_computational_memlet(block4, tasklet1, "_out", B, {symbolic::symbol("i")});
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(body.size(), 3);
}

TEST(BlockHoistingTest, Map_InvariantMemcpy) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant memcpy
    auto [block1, memcpy1] = stdlib::add_memcpy_block(builder, body, "C", "c", symbolic::integer(10), desc_ptr);

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& B = builder.add_access(block2, "B");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, B, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::symbol("i")});
    }

    // Loop invariant memcpy
    auto [block3, memcpy3] = stdlib::add_memcpy_block(builder, body, "C", "c", symbolic::integer(10), desc_ptr);

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, Map_InvariantMemmove) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant memmove
    auto [block1, memcpy1] = stdlib::add_memmove_block(builder, body, "C", "c", symbolic::integer(10), desc_ptr);

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& B = builder.add_access(block2, "B");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, B, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::symbol("i")});
    }

    // Loop invariant memmove
    auto [block3, memcpy3] = stdlib::add_memmove_block(builder, body, "C", "c", symbolic::integer(10), desc_ptr);

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, Map_InvariantMemset) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant memset
    auto [block1, memcpy1] =
        stdlib::add_memset_block(builder, body, "c", symbolic::integer(2), symbolic::integer(10), desc_ptr);

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::symbol("i")});
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(body.size(), 1);
}

TEST(BlockHoistingTest, For_InvariantMove) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // Loop invariant move
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder
            .add_dereference_memlet(block1, a, a_, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(BlockHoistingTest, For_InvariantView) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // Loop invariant view
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder.add_reference_memlet(block1, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(BlockHoistingTest, For_InvariantView_MultipleViews) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a1", opaque_ptr);
    builder.add_container("a2", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // Loop invariant view
    auto& block1 = builder.add_block(body);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a1");
        builder.add_reference_memlet(block1, a, a_, {symbolic::zero()}, desc_ptr);
    }
    auto& block2 = builder.add_block(body);
    {
        auto& a = builder.add_access(block2, "A");
        auto& a_ = builder.add_access(block2, "a2");
        builder.add_reference_memlet(block2, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Loop variant computation
    auto& block3 = builder.add_block(body);
    {
        auto& a1 = builder.add_access(block3, "a1");
        auto& a2 = builder.add_access(block3, "a2");
        auto& b = builder.add_access(block3, "B");
        auto& tasklet = builder.add_tasklet(block3, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block3, a1, tasklet, "_in1", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block3, a2, tasklet, "_in2", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block3, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(BlockHoistingTest, For_VariantAlloca) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    auto& block1 = builder.add_block(body);
    {
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<stdlib::AllocaNode>(block1, DebugInfo(), symbolic::one());
        builder.add_computational_memlet(block1, libnode, "_ret", c, {}, desc_ptr);
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::zero()});
    }
    auto& block3 = builder.add_block(body);
    {
        auto& B = builder.add_access(block3, "B");
        auto& A = builder.add_access(block3, "A");
        auto& tasklet1 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block3, B, tasklet1, "_in", {symbolic::symbol("i")});
        builder.add_computational_memlet(block3, tasklet1, "_out", A, {symbolic::symbol("i")});
    }
    auto& block4 = builder.add_block(body);
    {
        auto& c = builder.add_access(block4, "c");
        auto& B = builder.add_access(block4, "B");
        auto& tasklet1 = builder.add_tasklet(block4, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block4, c, tasklet1, "_in", {symbolic::zero()});
        builder.add_computational_memlet(block4, tasklet1, "_out", B, {symbolic::symbol("i")});
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(BlockHoistingTest, For_VariantMemcpy) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // C -> c memcpy
    auto [block1, memcpy1] = stdlib::add_memcpy_block(builder, body, "C", "c", symbolic::integer(10), desc_ptr);

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& B = builder.add_access(block2, "B");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, B, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::symbol("i")});
    }

    // Loop invariant memcpy
    auto [block3, memcpy3] = stdlib::add_memcpy_block(builder, body, "C", "c", symbolic::integer(10), desc_ptr);

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(BlockHoistingTest, For_VariantMemmove) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // C -> c memcpy
    auto [block1, memcpy1] = stdlib::add_memmove_block(builder, body, "C", "c", symbolic::integer(10), desc_ptr);

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& B = builder.add_access(block2, "B");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, B, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::symbol("i")});
    }

    // Loop invariant memmove
    auto [block3, memcpy3] = stdlib::add_memmove_block(builder, body, "C", "c", symbolic::integer(10), desc_ptr);

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(BlockHoistingTest, For_VariantMemset) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // 2 -> c memset
    auto [block1, memcpy1] =
        stdlib::add_memset_block(builder, body, "c", symbolic::integer(2), symbolic::integer(10), desc_ptr);
    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::symbol("i")});
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(BlockHoistingTest, IfElse_InvariantMove) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant move
    auto& block1 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder
            .add_dereference_memlet(block1, a, a_, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Case 1: Branch variant computation
    auto& block2 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::integer(3)}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::integer(3)}, desc_ptr);
    }

    // Case 2: Branch invariant move
    auto& block3 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block3, "A");
        auto& a_ = builder.add_access(block3, "a");
        builder
            .add_dereference_memlet(block3, a, a_, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Case 2: Branch variant computation
    auto& block4 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block4, "a");
        auto& b = builder.add_access(block4, "B");
        auto& tasklet = builder.add_tasklet(block4, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block4, a, tasklet, "_in", {symbolic::integer(8)}, desc_ptr);
        builder.add_computational_memlet(block4, tasklet, "_out", b, {symbolic::integer(8)}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}

TEST(BlockHoistingTest, IfElse_InvariantView) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant view
    auto& block1 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a");
        builder.add_reference_memlet(block1, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Case 1: Branch variant computation
    auto& block2 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "B");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, a, tasklet, "_in", {symbolic::integer(3)}, desc_ptr);
        builder.add_computational_memlet(block2, tasklet, "_out", b, {symbolic::integer(3)}, desc_ptr);
    }

    // Case 2: Branch invariant view
    auto& block3 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block3, "A");
        auto& a_ = builder.add_access(block3, "a");
        builder.add_reference_memlet(block3, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Case 2: Branch variant computation
    auto& block4 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block4, "a");
        auto& b = builder.add_access(block4, "B");
        auto& tasklet = builder.add_tasklet(block4, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block4, a, tasklet, "_in", {symbolic::integer(8)}, desc_ptr);
        builder.add_computational_memlet(block4, tasklet, "_out", b, {symbolic::integer(8)}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}

TEST(BlockHoistingTest, IfElse_InvariantView_MultipleViews) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a1", opaque_ptr);
    builder.add_container("a2", opaque_ptr);
    builder.add_container("B", opaque_ptr, true);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant views
    auto& block1 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block1, "A");
        auto& a_ = builder.add_access(block1, "a1");
        builder.add_reference_memlet(block1, a, a_, {symbolic::zero()}, desc_ptr);
    }
    auto& block2 = builder.add_block(case1);
    {
        auto& a = builder.add_access(block2, "A");
        auto& a_ = builder.add_access(block2, "a2");
        builder.add_reference_memlet(block2, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Case 1: Branch variant computation
    auto& block3 = builder.add_block(case1);
    {
        auto& a1 = builder.add_access(block3, "a1");
        auto& a2 = builder.add_access(block3, "a2");
        auto& b = builder.add_access(block3, "B");
        auto& tasklet = builder.add_tasklet(block3, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block3, a1, tasklet, "_in1", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block3, a2, tasklet, "_in2", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block3, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Case 2: Branch invariant views
    auto& block4 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block4, "A");
        auto& a_ = builder.add_access(block4, "a1");
        builder.add_reference_memlet(block4, a, a_, {symbolic::zero()}, desc_ptr);
    }
    auto& block5 = builder.add_block(case2);
    {
        auto& a = builder.add_access(block5, "A");
        auto& a_ = builder.add_access(block5, "a2");
        builder.add_reference_memlet(block5, a, a_, {symbolic::one()}, desc_ptr);
    }

    // Case 2: Branch variant computation
    auto& block6 = builder.add_block(case2);
    {
        auto& a1 = builder.add_access(block6, "a1");
        auto& a2 = builder.add_access(block6, "a2");
        auto& b = builder.add_access(block6, "B");
        auto& tasklet = builder.add_tasklet(block6, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block6, a1, tasklet, "_in1", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block6, a2, tasklet, "_in2", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block6, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}

TEST(BlockHoistingTest, IfElse_InvariantAlloca) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("c", desc_ptr);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant alloca
    {
        auto& block1 = builder.add_block(case1);
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<stdlib::AllocaNode>(block1, DebugInfo(), symbolic::one());
        builder.add_computational_memlet(block1, libnode, "_ret", c, {}, desc_ptr);
    }

    // Case 1: Branch variant computation
    {
        auto& block2 = builder.add_block(case1);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::zero()});
    }
    {
        auto& block3 = builder.add_block(case1);
        auto& B = builder.add_access(block3, "B");
        auto& A = builder.add_access(block3, "A");
        auto& tasklet1 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block3, B, tasklet1, "_in", {symbolic::integer(3)});
        builder.add_computational_memlet(block3, tasklet1, "_out", A, {symbolic::integer(3)});
    }
    {
        auto& block4 = builder.add_block(case1);
        auto& c = builder.add_access(block4, "c");
        auto& B = builder.add_access(block4, "B");
        auto& tasklet1 = builder.add_tasklet(block4, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block4, c, tasklet1, "_in", {symbolic::zero()});
        builder.add_computational_memlet(block4, tasklet1, "_out", B, {symbolic::integer(3)});
    }

    // Case 2: Branch invariant alloca
    {
        auto& block1 = builder.add_block(case2);
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<stdlib::AllocaNode>(block1, DebugInfo(), symbolic::one());
        builder.add_computational_memlet(block1, libnode, "_ret", c, {}, desc_ptr);
    }

    // Case 2: Branch variant computation
    {
        auto& block2 = builder.add_block(case2);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::zero()});
    }
    {
        auto& block3 = builder.add_block(case2);
        auto& B = builder.add_access(block3, "B");
        auto& A = builder.add_access(block3, "A");
        auto& tasklet1 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block3, B, tasklet1, "_in", {symbolic::integer(5)});
        builder.add_computational_memlet(block3, tasklet1, "_out", A, {symbolic::integer(5)});
    }
    {
        auto& block4 = builder.add_block(case2);
        auto& c = builder.add_access(block4, "c");
        auto& B = builder.add_access(block4, "B");
        auto& tasklet1 = builder.add_tasklet(block4, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block4, c, tasklet1, "_in", {symbolic::zero()});
        builder.add_computational_memlet(block4, tasklet1, "_out", B, {symbolic::integer(5)});
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(case1.size(), 3);
    EXPECT_EQ(case2.size(), 3);
}

TEST(BlockHoistingTest, IfElse_InvariantMemcpy) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant memcpy
    stdlib::add_memcpy_block(builder, case1, "C", "c", symbolic::integer(10), desc_ptr);

    // Case 1: Branch variant computation
    {
        auto& block2 = builder.add_block(case1);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(3)});
    }

    // Case 1: Branch invariant memcpy
    stdlib::add_memcpy_block(builder, case1, "C", "c", symbolic::integer(10), desc_ptr);

    // Case 2: Branch invariant memcpy
    stdlib::add_memcpy_block(builder, case2, "C", "c", symbolic::integer(10), desc_ptr);

    // Case 2: Branch variant computation
    {
        auto& block2 = builder.add_block(case2);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(5)});
    }

    // Case 2: Branch invariant memcpy
    stdlib::add_memcpy_block(builder, case2, "C", "c", symbolic::integer(10), desc_ptr);

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}

TEST(BlockHoistingTest, IfElse_InvariantMemmove) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant memcpy
    stdlib::add_memmove_block(builder, case1, "C", "c", symbolic::integer(10), desc_ptr);

    // Case 1: Branch variant computation
    {
        auto& block2 = builder.add_block(case1);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(3)});
    }

    // Case 1: Branch invariant memmove
    stdlib::add_memmove_block(builder, case1, "C", "c", symbolic::integer(10), desc_ptr);

    // Case 2: Branch invariant memcpy
    stdlib::add_memmove_block(builder, case2, "C", "c", symbolic::integer(10), desc_ptr);

    // Case 2: Branch variant computation
    {
        auto& block2 = builder.add_block(case2);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(5)});
    }

    // Case 2: Branch invariant memmove
    stdlib::add_memmove_block(builder, case2, "C", "c", symbolic::integer(10), desc_ptr);

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}

TEST(BlockHoistingTest, IfElse_InvariantMemset) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("c", desc_ptr);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant memcpy
    stdlib::add_memset_block(builder, case1, "c", symbolic::integer(2), symbolic::integer(10), desc_ptr);

    // Case 1: Branch variant computation
    {
        auto& block2 = builder.add_block(case1);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(3)});
    }

    // Case 2: Branch invariant memcpy
    stdlib::add_memset_block(builder, case2, "c", symbolic::integer(2), symbolic::integer(10), desc_ptr);

    // Case 2: Branch variant computation
    {
        auto& block2 = builder.add_block(case2);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(5)});
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}

TEST(BlockHoistingTest, Map_InvariantDataTransfers_CUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& map_stmt = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = map_stmt.root();

    // Loop invariant memcpy
    auto [block1, libnode1] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        body,
        "C",
        "c",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        desc_ptr,
        {},
        symbolic::integer(10),
        symbolic::zero()
    );

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& B = builder.add_access(block2, "B");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, B, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::symbol("i")});
    }

    // Loop invariant memcpy
    auto [block3, libnode3] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
        builder,
        body,
        "C",
        "c",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        desc_ptr,
        {},
        symbolic::integer(10),
        symbolic::zero()
    );

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(body.size(), 1);
}
/*
TEST(ExtendedBlockHoistingTest, For_InvariantDataTransfers_CUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);
    builder.add_container("i", sym_desc);

    auto& for_stmt = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = for_stmt.root();

    // Loop invariant memcpy
    auto& block1 = builder.add_block(body);
    {
        auto& C = builder.add_access(block1, "C");
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<cuda::CUDADataOffloadingNode>(
            block1,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            offloading::DataTransferDirection::H2D,
            offloading::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block1, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block1, libnode, "_dst", c, {}, desc_ptr);
    }

    // Loop variant computation
    auto& block2 = builder.add_block(body);
    {
        auto& A = builder.add_access(block2, "A");
        auto& B = builder.add_access(block2, "B");
        auto& c = builder.add_access(block2, "c");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, B, tasklet1, "_in2", {symbolic::symbol("i")});
        builder.add_computational_memlet(block2, tasklet1, "_out", c, {symbolic::symbol("i")});
    }

    // Loop invariant memcpy
    auto& block3 = builder.add_block(body);
    {
        auto& C = builder.add_access(block3, "C");
        auto& c = builder.add_access(block3, "c");
        auto& libnode = builder.add_library_node<cuda::CUDADataOffloadingNode>(
            block3,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            offloading::DataTransferDirection::D2H,
            offloading::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block3, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block3, libnode, "_dst", c, {}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(body.size(), 1);
}

TEST(ExtendedBlockHoistingTest, IfElse_InvariantDataTransfers_CUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("A", desc_ptr, true);
    builder.add_container("B", desc_ptr, true);
    builder.add_container("C", desc_ptr, true);
    builder.add_container("c", desc_ptr);

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Lt(symbolic::symbol("A"), symbolic::integer(5)));
    auto& case2 = builder.add_case(if_else, symbolic::Ge(symbolic::symbol("A"), symbolic::integer(5)));

    // Case 1: Branch invariant memcpy
    {
        auto& block1 = builder.add_block(case1);
        auto& C = builder.add_access(block1, "C");
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<cuda::CUDADataOffloadingNode>(
            block1,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            offloading::DataTransferDirection::H2D,
            offloading::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block1, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block1, libnode, "_dst", c, {}, desc_ptr);
    }

    // Case 1: Branch variant computation
    {
        auto& block2 = builder.add_block(case1);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(3)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(3)});
    }

    // Case 1: Branch invariant memcpy
    {
        auto& block3 = builder.add_block(case1);
        auto& C = builder.add_access(block3, "C");
        auto& c = builder.add_access(block3, "c");
        auto& libnode = builder.add_library_node<cuda::CUDADataOffloadingNode>(
            block3,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            offloading::DataTransferDirection::D2H,
            offloading::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block3, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block3, libnode, "_dst", c, {}, desc_ptr);
    }

    // Case 2: Branch invariant memcpy
    {
        auto& block1 = builder.add_block(case2);
        auto& C = builder.add_access(block1, "C");
        auto& c = builder.add_access(block1, "c");
        auto& libnode = builder.add_library_node<cuda::CUDADataOffloadingNode>(
            block1,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            offloading::DataTransferDirection::H2D,
            offloading::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block1, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block1, libnode, "_dst", c, {}, desc_ptr);
    }

    // Case 2: Branch variant computation
    {
        auto& block2 = builder.add_block(case2);
        auto& A = builder.add_access(block2, "A");
        auto& c = builder.add_access(block2, "c");
        auto& B = builder.add_access(block2, "B");
        auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block2, A, tasklet1, "_in1", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, c, tasklet1, "_in2", {symbolic::integer(5)});
        builder.add_computational_memlet(block2, tasklet1, "_out", B, {symbolic::integer(5)});
    }

    // Case 2: Branch invariant memcpy
    {
        auto& block3 = builder.add_block(case2);
        auto& C = builder.add_access(block3, "C");
        auto& c = builder.add_access(block3, "c");
        auto& libnode = builder.add_library_node<cuda::CUDADataOffloadingNode>(
            block3,
            DebugInfo(),
            symbolic::integer(10),
            symbolic::zero(),
            offloading::DataTransferDirection::D2H,
            offloading::BufferLifecycle::NO_CHANGE
        );
        builder.add_computational_memlet(block3, C, libnode, "_src", {}, desc_ptr);
        builder.add_computational_memlet(block3, libnode, "_dst", c, {}, desc_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
}
*/

TEST(BlockHoistingTest, waxpby_CUDA) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int32);
    builder.add_container("n", sym_desc, true);
    auto n = symbolic::symbol("n");
    auto id = symbolic::zero();
    builder.add_container("i", sym_desc);
    auto indvar = symbolic::symbol("i");
    auto condition = symbolic::Lt(indvar, n);
    auto init = symbolic::zero();
    auto update = symbolic::add(indvar, symbolic::one());

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("beta", base_desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("w", desc, true);
    builder.add_container("tmp", base_desc);

    types::Pointer d_desc(base_desc);
    d_desc.storage_type(types::StorageType::NV_Generic());
    const std::string d_x1 = cuda::CUDA_DEVICE_PREFIX + "x1";
    builder.add_container(d_x1, d_desc);
    const std::string d_x2 = cuda::CUDA_DEVICE_PREFIX + "x2";
    builder.add_container(d_x2, d_desc);
    const std::string d_x3 = cuda::CUDA_DEVICE_PREFIX + "x3";
    builder.add_container(d_x3, d_desc);
    const std::string d_y1 = cuda::CUDA_DEVICE_PREFIX + "y1";
    builder.add_container(d_y1, d_desc);
    const std::string d_y2 = cuda::CUDA_DEVICE_PREFIX + "y2";
    builder.add_container(d_y2, d_desc);
    const std::string d_y3 = cuda::CUDA_DEVICE_PREFIX + "y3";
    builder.add_container(d_y3, d_desc);
    const std::string d_w1 = cuda::CUDA_DEVICE_PREFIX + "w1";
    builder.add_container(d_w1, d_desc);
    const std::string d_w2 = cuda::CUDA_DEVICE_PREFIX + "w2";
    builder.add_container(d_w2, d_desc);
    const std::string d_w3 = cuda::CUDA_DEVICE_PREFIX + "w3";
    builder.add_container(d_w3, d_desc);

    types::Scalar cond_desc(types::PrimitiveType::Bool);
    builder.add_container("cond1", cond_desc);
    builder.add_container("cond2", cond_desc);

    {
        auto& block = builder.add_block(root);
        auto& alpha = builder.add_access(block, "alpha");
        auto& zero = builder.add_constant(block, "1.0", base_desc);
        auto& cond1 = builder.add_access(block, "cond1");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_oeq, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet, "_in1", {});
        builder.add_computational_memlet(block, zero, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", cond1, {});
    }

    {
        auto& block = builder.add_block(root);
        auto& beta = builder.add_access(block, "beta");
        auto& zero = builder.add_constant(block, "1.0", base_desc);
        auto& cond2 = builder.add_access(block, "cond2");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_oeq, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block, beta, tasklet, "_in1", {});
        builder.add_computational_memlet(block, zero, tasklet, "_in2", {});
        builder.add_computational_memlet(block, tasklet, "_out", cond2, {});
    }

    auto& if_else = builder.add_if_else(root);
    auto& case1 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("cond1"), symbolic::__true__()));
    auto& case2 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("cond2"), symbolic::__true__()));
    auto& case3 = builder.add_case(
        if_else,
        symbolic::
            And(symbolic::Ne(symbolic::symbol("cond1"), symbolic::__true__()),
                symbolic::Ne(symbolic::symbol("cond2"), symbolic::__true__()))
    );

    // Case 1

    {
        auto [block, libnode] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
            builder,
            case1,
            "x",
            d_x1,
            offloading::DataTransferDirection::H2D,
            offloading::BufferLifecycle::ALLOC,
            desc,
            {},
            n,
            id
        );
    }

    {
        auto [block, libnode] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
            builder,
            case1,
            "y",
            d_y1,
            offloading::DataTransferDirection::H2D,
            offloading::BufferLifecycle::ALLOC,
            desc,
            {},
            n,
            id
        );
    }

    {
        auto& map = builder.add_map(case1, indvar, condition, init, update, cuda::ScheduleType_CUDA::create());
        auto& block = builder.add_block(map.root());
        auto& d_w = builder.add_access(block, d_w1);
        auto& d_x = builder.add_access(block, d_x1);
        auto& beta = builder.add_access(block, "beta");
        auto& d_y = builder.add_access(block, d_y1);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, beta, tasklet, "_in1", {});
        builder.add_computational_memlet(block, d_y, tasklet, "_in2", {indvar});
        builder.add_computational_memlet(block, d_x, tasklet, "_in3", {indvar});
        builder.add_computational_memlet(block, tasklet, "_out", d_w, {indvar});
    }

    {
        auto [block, libnode] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
            builder,
            case1,
            "w",
            d_w1,
            offloading::DataTransferDirection::D2H,
            offloading::BufferLifecycle::FREE,
            desc,
            {},
            n,
            id
        );
    }

    // Case 2

    {
        auto [block, libnode] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
            builder,
            case2,
            "x",
            d_x2,
            offloading::DataTransferDirection::H2D,
            offloading::BufferLifecycle::ALLOC,
            desc,
            {},
            n,
            id
        );
    }

    {
        auto [block, libnode] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
            builder,
            case2,
            "y",
            d_y2,
            offloading::DataTransferDirection::H2D,
            offloading::BufferLifecycle::ALLOC,
            desc,
            {},
            n,
            id
        );
    }

    {
        auto& map = builder.add_map(case2, indvar, condition, init, update, cuda::ScheduleType_CUDA::create());
        auto& block = builder.add_block(map.root());
        auto& d_w = builder.add_access(block, d_w2);
        auto& alpha = builder.add_access(block, "alpha");
        auto& d_x = builder.add_access(block, d_x2);
        auto& d_y = builder.add_access(block, d_y2);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, alpha, tasklet, "_in1", {});
        builder.add_computational_memlet(block, d_x, tasklet, "_in2", {indvar});
        builder.add_computational_memlet(block, d_y, tasklet, "_in3", {indvar});
        builder.add_computational_memlet(block, tasklet, "_out", d_w, {indvar});
    }

    {
        auto [block, libnode] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
            builder,
            case2,
            "w",
            d_w2,
            offloading::DataTransferDirection::D2H,
            offloading::BufferLifecycle::FREE,
            desc,
            {},
            n,
            id
        );
    }

    // Case 3

    {
        auto [block, libnode] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
            builder,
            case3,
            "x",
            d_x3,
            offloading::DataTransferDirection::H2D,
            offloading::BufferLifecycle::ALLOC,
            desc,
            {},
            n,
            id
        );
    }

    {
        auto [block, libnode] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
            builder,
            case3,
            "y",
            d_y3,
            offloading::DataTransferDirection::H2D,
            offloading::BufferLifecycle::ALLOC,
            desc,
            {},
            n,
            id
        );
    }

    {
        auto& map = builder.add_map(case3, indvar, condition, init, update, cuda::ScheduleType_CUDA::create());
        auto& block = builder.add_block(map.root());
        auto& d_w = builder.add_access(block, d_w2);
        auto& alpha = builder.add_access(block, "alpha");
        auto& d_x = builder.add_access(block, d_x3);
        auto& beta = builder.add_access(block, "beta");
        auto& d_y = builder.add_access(block, d_y3);
        auto& tmp = builder.add_access(block, "tmp");
        auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(block, alpha, tasklet1, "_in1", {});
        builder.add_computational_memlet(block, d_x, tasklet1, "_in2", {indvar});
        builder.add_computational_memlet(block, tasklet1, "_out", tmp, {});
        auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
        builder.add_computational_memlet(block, beta, tasklet2, "_in1", {});
        builder.add_computational_memlet(block, d_y, tasklet2, "_in2", {indvar});
        builder.add_computational_memlet(block, tmp, tasklet2, "_in3", {});
        builder.add_computational_memlet(block, tasklet2, "_out", d_w, {indvar});
    }

    {
        auto [block, libnode] = offloading::add_offloading_block<cuda::CUDADataOffloadingNode>(
            builder,
            case3,
            "w",
            d_w2,
            offloading::DataTransferDirection::D2H,
            offloading::BufferLifecycle::FREE,
            desc,
            {},
            n,
            id
        );
    }

    dump_sdfg(sdfg, "0.init");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockHoistingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    dump_sdfg(sdfg, "1.hoisted");

    EXPECT_EQ(root.size(), 6);
    EXPECT_EQ(case1.size(), 1);
    EXPECT_EQ(case2.size(), 1);
    EXPECT_EQ(case3.size(), 1);
}
