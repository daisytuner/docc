#include "sdfg/analysis/escape_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/stdlib.h"
#include "sdfg/types/pointer.h"

using namespace sdfg;

TEST(EscapeAnalysisTest, MallocDetection_Transient) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("ptr", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "ptr");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    // Check that the malloc allocation is detected
    EXPECT_TRUE(escape_analysis.is_malloc_allocation("ptr"));
}

TEST(EscapeAnalysisTest, MallocDetection_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("arg0", opaque_desc, true); // true = argument

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "arg0");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_TRUE(escape_analysis.is_malloc_allocation("arg0"));
    EXPECT_TRUE(escape_analysis.escapes("arg0")); // Arguments escape
}

TEST(EscapeAnalysisTest, NonEscaping_TransientLocalUse) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("ptr", ptr_desc);

    // Block 1: malloc
    auto& block1 = builder.add_block(root);
    auto& malloc_output = builder.add_access(block1, "ptr");
    auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_output, {}, ptr_desc, DebugInfo());

    // Block 2: use the allocated memory (read/write)
    auto& block2 = builder.add_block(root);
    auto& read_access = builder.add_access(block2, "ptr");
    auto& write_access = builder.add_access(block2, "ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, read_access, tasklet, "_in", {symbolic::zero()}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", write_access, {symbolic::zero()}, ptr_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_TRUE(escape_analysis.is_malloc_allocation("ptr"));
    EXPECT_FALSE(escape_analysis.escapes("ptr"));

    auto non_escaping = escape_analysis.non_escaping_allocations();
    EXPECT_EQ(non_escaping.size(), 1);
    EXPECT_TRUE(non_escaping.count("ptr") > 0);
}

TEST(EscapeAnalysisTest, Escaping_ThroughReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("ptr", opaque_desc);

    // Block: malloc
    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "ptr");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    // Return the pointer
    builder.add_return(root, "ptr");

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_TRUE(escape_analysis.is_malloc_allocation("ptr"));
    EXPECT_TRUE(escape_analysis.escapes("ptr"));

    auto non_escaping = escape_analysis.non_escaping_allocations();
    EXPECT_EQ(non_escaping.size(), 0);
}

TEST(EscapeAnalysisTest, LastUse_SimpleCase) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("ptr", ptr_desc);

    // Block 1: malloc
    auto& block1 = builder.add_block(root);
    auto& malloc_output = builder.add_access(block1, "ptr");
    auto& malloc_node =
        builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(sizeof(int)));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_output, {}, ptr_desc, DebugInfo());

    // Block 2: use the allocated memory
    auto& block2 = builder.add_block(root);
    auto& last_access = builder.add_access(block2, "ptr");
    auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& constant = builder.add_constant(block2, "42", base_desc);
    builder.add_computational_memlet(block2, constant, tasklet, "_in", {}, base_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet, "_out", last_access, {symbolic::zero()}, ptr_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_FALSE(escape_analysis.escapes("ptr"));

    auto* last_use = escape_analysis.last_use("ptr");
    EXPECT_NE(last_use, nullptr);
    // The last use should be in block2
    EXPECT_EQ(last_use->container(), "ptr");
}

TEST(EscapeAnalysisTest, NoMalloc_NotDetected) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("x", desc);

    // Just a simple assignment
    auto sym_x = symbolic::symbol("x");
    builder.add_assignments(root, {{sym_x, symbolic::integer(42)}});

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_FALSE(escape_analysis.is_malloc_allocation("x"));
    EXPECT_FALSE(escape_analysis.escapes("x"));

    auto non_escaping = escape_analysis.non_escaping_allocations();
    EXPECT_EQ(non_escaping.size(), 0);
}

TEST(EscapeAnalysisTest, MultipleMallocs) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("ptr1", opaque_desc);
    builder.add_container("ptr2", opaque_desc);
    builder.add_container("arg_ptr", opaque_desc, true); // argument

    // Block: multiple mallocs
    auto& block = builder.add_block(root);

    auto& access1 = builder.add_access(block, "ptr1");
    auto& malloc1 = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, malloc1, "_ret", access1, {}, opaque_desc, DebugInfo());

    auto& access2 = builder.add_access(block, "ptr2");
    auto& malloc2 = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(2048));
    builder.add_computational_memlet(block, malloc2, "_ret", access2, {}, opaque_desc, DebugInfo());

    auto& access3 = builder.add_access(block, "arg_ptr");
    auto& malloc3 = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(512));
    builder.add_computational_memlet(block, malloc3, "_ret", access3, {}, opaque_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_TRUE(escape_analysis.is_malloc_allocation("ptr1"));
    EXPECT_TRUE(escape_analysis.is_malloc_allocation("ptr2"));
    EXPECT_TRUE(escape_analysis.is_malloc_allocation("arg_ptr"));

    EXPECT_FALSE(escape_analysis.escapes("ptr1"));
    EXPECT_FALSE(escape_analysis.escapes("ptr2"));
    EXPECT_TRUE(escape_analysis.escapes("arg_ptr")); // argument escapes

    auto non_escaping = escape_analysis.non_escaping_allocations();
    EXPECT_EQ(non_escaping.size(), 2);
    EXPECT_TRUE(non_escaping.count("ptr1") > 0);
    EXPECT_TRUE(non_escaping.count("ptr2") > 0);
}

TEST(EscapeAnalysisTest, MallocInLoop_NonEscaping) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar idx_desc(types::PrimitiveType::UInt64);
    types::Pointer opaque_desc;
    builder.add_container("i", idx_desc);
    builder.add_container("ptr", opaque_desc);

    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, symbolic::integer(10));
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Block inside loop: malloc and use
    auto& block = builder.add_block(body);
    auto& access_node = builder.add_access(block, "ptr");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_TRUE(escape_analysis.is_malloc_allocation("ptr"));
    EXPECT_FALSE(escape_analysis.escapes("ptr"));
}

TEST(EscapeAnalysisTest, LastUse_NullForEscaping) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Pointer opaque_desc;
    builder.add_container("ptr", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node = builder.add_access(block, "ptr");
    auto& lib_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), symbolic::integer(1024));
    builder.add_computational_memlet(block, lib_node, "_ret", access_node, {}, opaque_desc, DebugInfo());

    // Return the pointer - makes it escape
    builder.add_return(root, "ptr");

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_TRUE(escape_analysis.escapes("ptr"));
    EXPECT_EQ(escape_analysis.last_use("ptr"), nullptr);
}

TEST(EscapeAnalysisTest, LastUse_NullForNonMalloc) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("x", desc);

    auto sym_x = symbolic::symbol("x");
    builder.add_assignments(root, {{sym_x, symbolic::integer(42)}});

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_EQ(escape_analysis.last_use("x"), nullptr);
}

TEST(EscapeAnalysisTest, Escapes_FalseForNonMalloc) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("x", desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    // Non-malloc containers return false for escapes()
    EXPECT_FALSE(escape_analysis.escapes("x"));
    EXPECT_FALSE(escape_analysis.escapes("nonexistent"));
}

TEST(EscapeAnalysisTest, LastUse_SequentialBlocks) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Pointer ptr_desc(base_desc);
    builder.add_container("ptr", ptr_desc);
    builder.add_container("result", base_desc);

    // Block 1: malloc
    auto& block1 = builder.add_block(root);
    auto& malloc_output = builder.add_access(block1, "ptr");
    auto& malloc_node =
        builder.add_library_node<stdlib::MallocNode>(block1, DebugInfo(), symbolic::integer(sizeof(int) * 10));
    builder.add_computational_memlet(block1, malloc_node, "_ret", malloc_output, {}, ptr_desc, DebugInfo());

    // Block 2: write to allocated memory
    auto& block2 = builder.add_block(root);
    auto& write_ptr = builder.add_access(block2, "ptr");
    auto& constant1 = builder.add_constant(block2, "100", base_desc);
    auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, constant1, tasklet1, "_in", {}, base_desc, DebugInfo());
    builder.add_computational_memlet(block2, tasklet1, "_out", write_ptr, {symbolic::integer(0)}, ptr_desc, DebugInfo());

    // Block 3: read from allocated memory (this should be the last use)
    auto& block3 = builder.add_block(root);
    auto& read_ptr = builder.add_access(block3, "ptr");
    auto& result_access = builder.add_access(block3, "result");
    auto& tasklet2 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block3, read_ptr, tasklet2, "_in", {symbolic::integer(0)}, ptr_desc, DebugInfo());
    builder.add_computational_memlet(block3, tasklet2, "_out", result_access, {}, base_desc, DebugInfo());

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& escape_analysis = analysis_manager.get<analysis::EscapeAnalysis>();

    EXPECT_TRUE(escape_analysis.is_malloc_allocation("ptr"));
    EXPECT_FALSE(escape_analysis.escapes("ptr"));

    auto* last_use = escape_analysis.last_use("ptr");
    EXPECT_NE(last_use, nullptr);
    EXPECT_EQ(last_use->container(), "ptr");
}
