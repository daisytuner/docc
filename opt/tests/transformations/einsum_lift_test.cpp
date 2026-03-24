#include "sdfg/transformations/einsum_lift.h"

#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(EinsumLiftTest, NoReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& c_out = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumLift transformation(tasklet);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumLiftTest, Simple_fp_add) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& a_out = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumLift transformation(tasklet);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 3);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);

    auto* libnode = *dfg.library_nodes().begin();
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in2", "__einsum_out"}));
}

TEST(EinsumLiftTest, Simple_int_add) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& a_out = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, b_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumLift transformation(tasklet);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumLiftTest, Simple_fp_fma) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& constant_two = builder.add_constant(block, "2.0", desc);
    auto& a_out = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"});
    builder.add_computational_memlet(block, constant_two, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, a_in, tasklet, "_in3", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumLift transformation(tasklet);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 4);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);

    auto* libnode = *dfg.library_nodes().begin();
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in1", "_in2", "__einsum_out"}));
}

TEST(EinsumLiftTest, Simple_fp_sub) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& a_out = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumLift transformation(tasklet);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());

    auto& dfg = block.dataflow();
    EXPECT_EQ(dfg.data_nodes().size(), 4);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);

    auto* libnode = *dfg.library_nodes().begin();
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"__einsum_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in2", "__einsum_const", "__einsum_out"}));
}

TEST(EinsumLiftTest, WrongSubtractionOrder) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& a_out = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, b_in, tasklet, "_in1", {});
    builder.add_computational_memlet(block, a_in, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {});

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumLift transformation(tasklet);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(EinsumLiftTest, NonEqualSubsets) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(root);
    auto& a_in = builder.add_access(block, "a");
    auto& b_in = builder.add_access(block, "b");
    auto& a_out = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::zero()});
    builder.add_computational_memlet(block, b_in, tasklet, "_in2", {symbolic::zero()});
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::one()});

    analysis::AnalysisManager analysis_manager(sdfg);
    transformations::EinsumLift transformation(tasklet);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}
