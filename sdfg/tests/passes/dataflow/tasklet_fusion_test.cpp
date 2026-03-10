#include "sdfg/passes/dataflow/tasklet_fusion.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(TaskletFusionTest, SimpleInAssign) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);

    // Add block with two tasklets
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, a, tasklet1, "_in", {});
    builder.add_computational_memlet(block, tasklet1, "_out", b, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, b, tasklet2, "_in1", {});
    builder.add_computational_memlet(block, c, tasklet2, "_in2", {});
    builder.add_computational_memlet(block, tasklet2, "_out", d, {});

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::TaskletFusionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check
    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.nodes().size(), 4);
    EXPECT_EQ(dfg.tasklets(), std::unordered_set<data_flow::Tasklet*>({&tasklet2}));
    EXPECT_EQ(dfg.data_nodes(), std::unordered_set<data_flow::AccessNode*>({&a, &c, &d}));
}

TEST(TaskletFusionTest, InAssignWithExtraRead) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);
    builder.add_container("e", desc);

    // Add block with two tasklets
    auto& block1 = builder.add_block(root);
    auto& a = builder.add_access(block1, "a");
    auto& b1 = builder.add_access(block1, "b");
    auto& c = builder.add_access(block1, "c");
    auto& d = builder.add_access(block1, "d");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block1, a, tasklet1, "_in", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", b1, {});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block1, b1, tasklet2, "_in1", {});
    builder.add_computational_memlet(block1, c, tasklet2, "_in2", {});
    builder.add_computational_memlet(block1, tasklet2, "_out", d, {});

    // Add block with extra read
    auto& block2 = builder.add_block(root);
    auto& b2 = builder.add_access(block2, "b");
    auto& e = builder.add_access(block2, "e");
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block2, b2, tasklet3, "_in", {});
    builder.add_computational_memlet(block2, tasklet3, "_out", e, {});

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::TaskletFusionPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(TaskletFusionTest, InCasting) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", sym_desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);

    // Add block with two tasklets
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, a, tasklet1, "_in", {});
    builder.add_computational_memlet(block, tasklet1, "_out", b, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, b, tasklet2, "_in1", {});
    builder.add_computational_memlet(block, c, tasklet2, "_in2", {});
    builder.add_computational_memlet(block, tasklet2, "_out", d, {});

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::TaskletFusionPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(TaskletFusionTest, SimpleInAssignWithAccessNodeMerging) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    // Add block with two tasklets
    auto& block = builder.add_block(root);
    auto& a1 = builder.add_access(block, "a");
    auto& a2 = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, a1, tasklet1, "_in", {});
    builder.add_computational_memlet(block, tasklet1, "_out", b, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, b, tasklet2, "_in1", {});
    builder.add_computational_memlet(block, a2, tasklet2, "_in2", {});
    builder.add_computational_memlet(block, tasklet2, "_out", c, {});

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::TaskletFusionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check
    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.nodes().size(), 3);
}

TEST(TaskletFusionTest, ComplexInAssignWithAccessNodeMerging) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);

    // Add block with two tasklets
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, a, tasklet1, "_in", {});
    builder.add_computational_memlet(block, tasklet1, "_out", b, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, a, tasklet2, "_in", {});
    builder.add_computational_memlet(block, tasklet2, "_out", c, {});
    auto& tasklet3 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, b, tasklet3, "_in1", {});
    builder.add_computational_memlet(block, c, tasklet3, "_in2", {});
    builder.add_computational_memlet(block, tasklet3, "_out", d, {});

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::TaskletFusionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check
    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.nodes().size(), 3);
    EXPECT_EQ(dfg.tasklets(), std::unordered_set<data_flow::Tasklet*>({&tasklet3}));
    EXPECT_EQ(dfg.data_nodes(), std::unordered_set<data_flow::AccessNode*>({&a, &d}));
}

TEST(TaskletFusionTest, SimpleOutAssign) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);

    // Add block with two tasklets
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet1, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet1, "_in2", {});
    builder.add_computational_memlet(block, tasklet1, "_out", c, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, c, tasklet2, "_in", {});
    builder.add_computational_memlet(block, tasklet2, "_out", d, {});

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::TaskletFusionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    // Check
    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.nodes().size(), 4);
    EXPECT_EQ(dfg.tasklets(), std::unordered_set<data_flow::Tasklet*>({&tasklet1}));
    EXPECT_EQ(dfg.data_nodes(), std::unordered_set<data_flow::AccessNode*>({&a, &b, &d}));
}

TEST(TaskletFusionTest, OutAssignWithExtraRead) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    builder.add_container("d", desc);
    builder.add_container("e", desc);

    // Add block with two tasklets
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c1 = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet1, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet1, "_in2", {});
    builder.add_computational_memlet(block, tasklet1, "_out", c1, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, c1, tasklet2, "_in", {});
    builder.add_computational_memlet(block, tasklet2, "_out", d, {});

    // Add block with extra read
    auto& block2 = builder.add_block(root);
    auto& c2 = builder.add_access(block2, "c");
    auto& e = builder.add_access(block2, "e");
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block2, c2, tasklet3, "_in", {});
    builder.add_computational_memlet(block2, tasklet3, "_out", e, {});

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::TaskletFusionPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(TaskletFusionTest, OutCasting) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", sym_desc);
    builder.add_container("b", sym_desc);
    builder.add_container("c", sym_desc);
    builder.add_container("d", desc);

    // Add block with two tasklets
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::int_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet1, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet1, "_in2", {});
    builder.add_computational_memlet(block, tasklet1, "_out", c, {});
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(block, c, tasklet2, "_in", {});
    builder.add_computational_memlet(block, tasklet2, "_out", d, {});

    // Apply pass
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::TaskletFusionPass pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}
