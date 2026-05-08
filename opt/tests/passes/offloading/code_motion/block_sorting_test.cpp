#include "sdfg/passes/offloading/code_motion/block_sorting.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(BlockSortingTest, Compute_Before_Move) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr, true);
    builder.add_container("b", opaque_ptr, true);
    builder.add_container("B", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    // Computation block
    auto& block1 = builder.add_block(root);
    {
        auto& a = builder.add_access(block1, "a");
        auto& b = builder.add_access(block1, "b");
        auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block1, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block1, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Move block
    auto& block2 = builder.add_block(root);
    {
        auto& a = builder.add_access(block2, "A");
        auto& b = builder.add_access(block2, "B");
        builder.add_dereference_memlet(block2, a, b, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockSortingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(&root.at(0).first, &block2);
    EXPECT_EQ(&root.at(1).first, &block1);
}

TEST(BlockSortingTest, Compute_Before_Malloc) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr, true);
    builder.add_container("b", opaque_ptr, true);
    builder.add_container("i", sym_desc);

    // Computation block
    auto& block1 = builder.add_block(root);
    {
        auto& a = builder.add_access(block1, "a");
        auto& b = builder.add_access(block1, "b");
        auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(block1, a, tasklet, "_in", {symbolic::symbol("i")}, desc_ptr);
        builder.add_computational_memlet(block1, tasklet, "_out", b, {symbolic::symbol("i")}, desc_ptr);
    }

    // Malloc block
    auto& block2 = builder.add_block(root);
    {
        auto& a = builder.add_access(block2, "A");
        auto& libnode = builder.add_library_node<stdlib::MallocNode>(block2, DebugInfo(), symbolic::integer(100));
        builder.add_computational_memlet(block2, libnode, "_ret", a, {}, opaque_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockSortingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(&root.at(0).first, &block2);
    EXPECT_EQ(&root.at(1).first, &block1);
}

TEST(BlockSortingTest, Move_Before_Malloc) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr, true);
    builder.add_container("b", opaque_ptr, true);

    // Move block
    auto& block1 = builder.add_block(root);
    {
        auto& a = builder.add_access(block1, "a");
        auto& b = builder.add_access(block1, "b");
        builder.add_dereference_memlet(block1, a, b, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Malloc block
    auto& block2 = builder.add_block(root);
    {
        auto& a = builder.add_access(block2, "A");
        auto& libnode = builder.add_library_node<stdlib::MallocNode>(block2, DebugInfo(), symbolic::integer(100));
        builder.add_computational_memlet(block2, libnode, "_ret", a, {}, opaque_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockSortingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(&root.at(0).first, &block2);
    EXPECT_EQ(&root.at(1).first, &block1);
}

TEST(BlockSortingTest, Free_Before_Move) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr, true);
    builder.add_container("B", opaque_ptr, true);

    // Free block
    auto& block1 = builder.add_block(root);
    {
        auto& a_in = builder.add_access(block1, "a");
        auto& libnode = builder.add_library_node<stdlib::FreeNode>(block1, DebugInfo());
        builder.add_computational_memlet(block1, a_in, libnode, "_ptr", {}, desc_ptr);
    }

    // Move block
    auto& block2 = builder.add_block(root);
    {
        auto& a = builder.add_access(block2, "A");
        auto& b = builder.add_access(block2, "B");
        builder.add_dereference_memlet(block2, a, b, true, types::Pointer(static_cast<const types::IType&>(opaque_ptr)));
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockSortingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(&root.at(0).first, &block2);
    EXPECT_EQ(&root.at(1).first, &block1);
}

TEST(BlockSortingTest, Free_Before_Malloc) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_ptr;
    builder.add_container("A", opaque_ptr, true);
    builder.add_container("a", opaque_ptr, true);

    // Free block
    auto& block1 = builder.add_block(root);
    {
        auto& a_in = builder.add_access(block1, "a");
        auto& libnode = builder.add_library_node<stdlib::FreeNode>(block1, DebugInfo());
        builder.add_computational_memlet(block1, a_in, libnode, "_ptr", {}, desc_ptr);
    }

    // Malloc block
    auto& block2 = builder.add_block(root);
    {
        auto& a = builder.add_access(block2, "A");
        auto& libnode = builder.add_library_node<stdlib::MallocNode>(block2, DebugInfo(), symbolic::integer(100));
        builder.add_computational_memlet(block2, libnode, "_ret", a, {}, opaque_ptr);
    }

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::BlockSortingPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(&root.at(0).first, &block2);
    EXPECT_EQ(&root.at(1).first, &block1);
}
