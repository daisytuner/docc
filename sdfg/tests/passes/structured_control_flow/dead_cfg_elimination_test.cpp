#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(DeadCFGEliminationTest, Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::DeadCFGElimination dce_pass;
    EXPECT_FALSE(dce_pass.run(builder, analysis_manager));
}

TEST(DeadCFGEliminationTest, AssignmentsAfterReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type, true);

    auto& root = builder.subject().root();
    auto& return_node = builder.add_return(root, "i");
    auto& assignments = builder.add_assignments(root, {{sdfg::symbolic::symbol("i"), sdfg::symbolic::integer(10)}});

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(dyn_cast<AssignmentBlock&>(root.at(1)).assignments().size(), 1);

    dump_sdfg(builder.subject(), "0.before");

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    EXPECT_EQ(root.size(), 1);
    EXPECT_TRUE(dyn_cast<Return*>(&root.at(0)) != nullptr);
}

TEST(DeadCFGEliminationTest, NodesAfterReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type, true);

    auto& root = builder.subject().root();
    auto& return_node = builder.add_return(root, "i");
    auto& assignments = builder.add_assignments(root, {{sdfg::symbolic::symbol("i"), sdfg::symbolic::integer(10)}});
    auto& block = builder.add_block(root);

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(dyn_cast<AssignmentBlock&>(root.at(1)).assignments().size(), 1);

    dump_sdfg(builder.subject(), "0.before");

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    EXPECT_EQ(root.size(), 1);
    EXPECT_TRUE(dyn_cast<Return*>(&root.at(0)) != nullptr);
}

TEST(DeadCFGEliminationTest, TrivialNonSequentialMapIsKept) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("i", int_type);
    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& map_loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType("TEST_OFFLOAD", structured_control_flow::ScheduleTypeCategory::Offloader)
    );
    builder.add_block(map_loop.root());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    dce_pass.run(builder, analysis_manager);

    ASSERT_EQ(root.size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Map*>(&root.at(0)) != nullptr);
}

TEST(DeadCFGEliminationTest, NonTrivialMap) {
    // Test non-trivial loop: for (i = 0; i < 10; i++) - should NOT be eliminated
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    // for (i = 0; i < 10; i++)
    auto indvar = symbolic::symbol("i");
    auto& for_loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Add a block in the loop body
    auto& body_block = builder.add_block(for_loop.root());

    EXPECT_EQ(root.size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Map*>(&root.at(0)) != nullptr);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    dce_pass.run(builder, analysis_manager);

    EXPECT_EQ(root.size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Map*>(&root.at(0)) != nullptr);
}

TEST(DeadCFGEliminationTest, TrivialSequentialMapIsKept) {
    // Trivial maps are eliminated by the TrivialLoopElimination transformation, not by this pass
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("i", int_type);
    auto& root = builder.subject().root();

    auto indvar = symbolic::symbol("i");
    auto& map_loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(map_loop.root());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    dce_pass.run(builder, analysis_manager);

    ASSERT_EQ(root.size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Map*>(&root.at(0)) != nullptr);
}

TEST(DeadCFGEliminationTest, ZeroIterationsMap) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& map = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::zero()),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(map.root());
    auto& a_access = builder.add_access(block, "a");
    auto& b_access = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_access, tasklet, "_in", {indvar});
    builder.add_computational_memlet(block, tasklet, "_out", b_access, {indvar});

    ASSERT_NO_THROW(sdfg.validate());

    dump_sdfg(builder.subject(), "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    ASSERT_EQ(root.size(), 0);
}

TEST(DeadCFGEliminationTest, ZeroIterationsReduction) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    builder.add_container("i", sym_desc);

    auto& block1 = builder.add_block(root);
    auto& constant_zero = builder.add_constant(block1, "0.0f", base_desc);
    auto& b_access = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, constant_zero, tasklet1, "_in", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", b_access, {});

    auto indvar = symbolic::symbol("i");
    auto& reduce = builder.add_reduce(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::zero()),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        {{.operation = ReductionOperation::Add, .container = "b"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& block2 = builder.add_block(reduce.root());
    auto& a_access = builder.add_access(block2, "a");
    auto& b_access_in = builder.add_access(block2, "b");
    auto& b_access_out = builder.add_access(block2, "b");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block2, b_access_in, tasklet2, "_in1", {});
    builder.add_computational_memlet(block2, a_access, tasklet2, "_in2", {indvar});
    builder.add_computational_memlet(block2, tasklet2, "_out", b_access_out, {});

    ASSERT_NO_THROW(sdfg.validate());

    dump_sdfg(builder.subject(), "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    ASSERT_EQ(root.size(), 1);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&root.at(0)), nullptr);
}

TEST(DeadCFGEliminationTest, TrivialBranch) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("a", base_desc);
    builder.add_container("b", base_desc);

    auto& if_else = builder.add_if_else(root);
    auto& trivial_case = builder.add_case(if_else, symbolic::__true__());

    auto& block = builder.add_block(trivial_case);
    auto& a_access = builder.add_access(block, "a");
    auto& b_access = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_access, {});

    ASSERT_NO_THROW(sdfg.validate());

    dump_sdfg(builder.subject(), "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    ASSERT_EQ(root.size(), 1);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&root.at(0)), nullptr);
}

TEST(DeadCFGEliminationTest, FalseBranch) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("a", base_desc);
    builder.add_container("b", base_desc);

    auto& if_else = builder.add_if_else(root);
    auto& trivial_case = builder.add_case(if_else, symbolic::__false__());

    auto& block = builder.add_block(trivial_case);
    auto& a_access = builder.add_access(block, "a");
    auto& b_access = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_access, {});

    ASSERT_NO_THROW(sdfg.validate());

    dump_sdfg(builder.subject(), "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    ASSERT_EQ(root.size(), 1);
    auto* new_if_else = dyn_cast<structured_control_flow::IfElse*>(&root.at(0));
    ASSERT_NE(new_if_else, nullptr);
    EXPECT_EQ(new_if_else, &if_else);
    EXPECT_EQ(new_if_else->size(), 0);
}

TEST(DeadCFGEliminationTest, TrailingDeadBranch) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("a", base_desc);
    builder.add_container("b", base_desc);

    auto& if_else = builder.add_if_else(root);
    auto& branch_false = builder.add_case(if_else, symbolic::__false__());
    auto& branch_legit = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("a"), symbolic::one()));
    auto& branch_dead = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("a"), symbolic::zero()));

    {
        auto& block = builder.add_block(branch_false);
        auto& a_access = builder.add_access(block, "a");
        auto& b_access = builder.add_access(block, "b");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_access, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", b_access, {});
    }

    {
        auto& block = builder.add_block(branch_legit);
        auto& a_access = builder.add_access(block, "a");
        auto& b_access = builder.add_access(block, "b");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_access, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", b_access, {});
    }

    ASSERT_NO_THROW(sdfg.validate());

    dump_sdfg(builder.subject(), "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    ASSERT_EQ(root.size(), 1);
    auto* new_if_else = dyn_cast<structured_control_flow::IfElse*>(&root.at(0));
    ASSERT_NE(new_if_else, nullptr);
    EXPECT_EQ(new_if_else, &if_else);
    EXPECT_EQ(new_if_else->size(), 1);
}

TEST(DeadCFGEliminationTest, IfElseToIfConversionFirst) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("a", base_desc);
    builder.add_container("b", base_desc);

    auto& if_else = builder.add_if_else(root);
    auto cond = symbolic::Eq(symbolic::symbol("a"), symbolic::one());
    auto& first_case = builder.add_case(if_else, cond);
    auto& second_case = builder.add_case(if_else, symbolic::Not(cond));

    auto& block = builder.add_block(first_case);
    auto& a_access = builder.add_access(block, "a");
    auto& b_access = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_access, {});

    ASSERT_NO_THROW(sdfg.validate());

    dump_sdfg(builder.subject(), "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    ASSERT_EQ(root.size(), 1);
    auto* new_if_else = dyn_cast<structured_control_flow::IfElse*>(&root.at(0));
    ASSERT_NE(new_if_else, nullptr);
    EXPECT_EQ(new_if_else, &if_else);
    EXPECT_EQ(new_if_else->size(), 1);
}

TEST(DeadCFGEliminationTest, IfElseToIfConversionSecond) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("a", base_desc);
    builder.add_container("b", base_desc);

    auto& if_else = builder.add_if_else(root);
    auto cond = symbolic::Eq(symbolic::symbol("a"), symbolic::one());
    auto& first_case = builder.add_case(if_else, cond);
    auto& second_case = builder.add_case(if_else, symbolic::Not(cond));

    auto& block = builder.add_block(second_case);
    auto& a_access = builder.add_access(block, "a");
    auto& b_access = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", b_access, {});

    ASSERT_NO_THROW(sdfg.validate());

    dump_sdfg(builder.subject(), "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.after");

    ASSERT_EQ(root.size(), 1);
    auto* new_if_else = dyn_cast<structured_control_flow::IfElse*>(&root.at(0));
    ASSERT_NE(new_if_else, nullptr);
    EXPECT_EQ(new_if_else, &if_else);
    EXPECT_EQ(new_if_else->size(), 1);
}
