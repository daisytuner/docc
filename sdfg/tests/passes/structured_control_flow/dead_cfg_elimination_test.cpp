#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/types/pointer.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(DeadCFGEliminationTest, AssignmentsAfterReturn) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type, true);

    auto& root = builder.subject().root();
    auto& return_node = builder.add_return(root, "i");
    auto& assignments = builder.add_assignments(root, {{sdfg::symbolic::symbol("i"), sdfg::symbolic::integer(10)}});

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(dyn_cast<AssignmentBlock&>(root.at(1)).assignments().size(), 1);

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

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

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

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
