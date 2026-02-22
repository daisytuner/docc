#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

#include <gtest/gtest.h>

#include <ostream>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"

using namespace sdfg;

TEST(DeadCFGEliminationTest, AssignmentsAfterReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type, true);

    auto& root = builder.subject().root();
    auto& return_node = builder.add_return(root, "i", {{sdfg::symbolic::symbol("i"), sdfg::symbolic::integer(10)}});
    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(root.at(0).second.size(), 1);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(root.at(0).second.size(), 0);
}

TEST(DeadCFGEliminationTest, NodesAfterReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type, true);

    auto& root = builder.subject().root();
    auto& return_node = builder.add_return(root, "i", {{sdfg::symbolic::symbol("i"), sdfg::symbolic::integer(10)}});
    auto& block = builder.add_block(root);

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(root.at(0).second.size(), 1);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(root.at(0).second.size(), 0);
}

TEST(DeadCFGEliminationTest, TrivialMap) {
    // Test trivial map: map (i = 0; i < 1; i++) with body
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::UInt64);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    // map (i = 0; i < 1; i++)
    auto indvar = symbolic::symbol("i");
    auto& map_loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Add a block in the loop body
    auto& body_block = builder.add_block(map_loop.root());

    EXPECT_EQ(root.size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Map*>(&root.at(0).first) != nullptr);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 0);
}

TEST(DeadCFGEliminationTest, TrivialMapWithSymbolicInit) {
    // Test trivial loop: for (i = N; i < N+1; i++) with body
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::UInt64);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    // for (i = N; i < N + 1; i++)
    auto indvar = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    auto init = N;
    auto bound = symbolic::add(N, symbolic::integer(1));

    auto& for_loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, bound),
        init,
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Add a block in the loop body
    auto& body_block = builder.add_block(for_loop.root());

    EXPECT_EQ(root.size(), 1);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 0);
}

TEST(DeadCFGEliminationTest, NonTrivialMap) {
    // Test non-trivial loop: for (i = 0; i < 10; i++) - should NOT be eliminated
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::UInt64);
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
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Map*>(&root.at(0).first) != nullptr);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    dce_pass.run(builder, analysis_manager);

    EXPECT_EQ(root.size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Map*>(&root.at(0).first) != nullptr);
}

TEST(DeadCFGEliminationTest, TrivialMapWithLessThanOrEqual) {
    // Test trivial loop with <=: for (i = 0; i <= 0; i++)
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::UInt64);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    // for (i = 0; i <= 0; i++) - executes once (bound becomes 0+1=1, trip count = 1-0=1)
    auto indvar = symbolic::symbol("i");
    auto& for_loop = builder.add_map(
        root,
        indvar,
        symbolic::Le(indvar, symbolic::integer(0)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Add a block in the loop body
    auto& body_block = builder.add_block(for_loop.root());

    EXPECT_EQ(root.size(), 1);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    EXPECT_EQ(root.size(), 0);
}

TEST(DeadCFGEliminationTest, TrivialMapEmptyBody) {
    // Test trivial loop with empty body: for (i = 0; i < 1; i++) {}
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::UInt64);
    builder.add_container("i", int_type);

    auto& root = builder.subject().root();

    // for (i = 0; i < 1; i++) with empty body
    auto indvar = symbolic::symbol("i");
    auto& for_loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(for_loop.root().size(), 0);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    // Trivial loop with empty body should be removed entirely
    EXPECT_EQ(root.size(), 0);
}
