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

TEST(DeadCFGEliminationTest, TrivialMap) {
    // for (j = 0; j < N; j++) { map (i = 0; i < 1; i++) { for (k = 0; k < N; k++) {} } }
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    builder.add_container("k", int_type);

    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");

    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    auto indvar = symbolic::symbol("i");
    auto& map_loop = builder.add_map(
        outer_loop.root(),
        indvar,
        symbolic::Lt(indvar, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto k = symbolic::symbol("k");
    auto& inner_loop =
        builder
            .add_for(map_loop.root(), k, symbolic::Lt(k, N), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    ASSERT_EQ(root.size(), 1);
    EXPECT_EQ(&root.at(0), &outer_loop);
    ASSERT_EQ(outer_loop.root().size(), 1);
    EXPECT_EQ(&outer_loop.root().at(0), &inner_loop);
}

TEST(DeadCFGEliminationTest, ConsecutiveTrivialMaps) {
    // for (j) { map (i0 = 0; i0 < 1) { map (i1 = 0; i1 < 1) { for (k) {} } } }
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("N", int_type, true);
    builder.add_container("i0", int_type);
    builder.add_container("i1", int_type);
    builder.add_container("j", int_type);
    builder.add_container("k", int_type);

    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");

    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    auto i0 = symbolic::symbol("i0");
    auto& map_0 = builder.add_map(
        outer_loop.root(),
        i0,
        symbolic::Lt(i0, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(i0, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto i1 = symbolic::symbol("i1");
    auto& map_1 = builder.add_map(
        map_0.root(),
        i1,
        symbolic::Lt(i1, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(i1, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto k = symbolic::symbol("k");
    auto& inner_loop =
        builder
            .add_for(map_1.root(), k, symbolic::Lt(k, N), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    ASSERT_EQ(root.size(), 1);
    EXPECT_EQ(&root.at(0), &outer_loop);
    ASSERT_EQ(outer_loop.root().size(), 1);
    EXPECT_EQ(&outer_loop.root().at(0), &inner_loop);
}

TEST(DeadCFGEliminationTest, TrivialMapWithoutLoopNeighborsIsKept) {
    // map (i = 0; i < 1; i++) { <block> } followed by a dead block
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
    builder.add_block(root);

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    // Both empty blocks are removed, but the map itself is kept.
    ASSERT_EQ(root.size(), 1);
    EXPECT_EQ(&root.at(0), &map_loop);
    EXPECT_EQ(map_loop.root().size(), 0);
}

TEST(DeadCFGEliminationTest, OutermostTrivialMap) {
    // map (i = 0; i < 1; i++) { for (k = 0; k < N; k++) {} }
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    builder.add_container("k", int_type);

    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");

    auto indvar = symbolic::symbol("i");
    auto& map_loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto k = symbolic::symbol("k");
    auto& inner_loop =
        builder
            .add_for(map_loop.root(), k, symbolic::Lt(k, N), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    ASSERT_EQ(root.size(), 1);
    EXPECT_EQ(&root.at(0), &inner_loop);
}

TEST(DeadCFGEliminationTest, InnermostTrivialMap) {
    // for (j = 0; j < N; j++) { map (i = 0; i < 1; i++) { <block> } }
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    types::Pointer int_pointer_type(int_type);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    builder.add_container("a", int_pointer_type);

    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");

    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    auto indvar = symbolic::symbol("i");
    auto& map_loop = builder.add_map(
        outer_loop.root(),
        indvar,
        symbolic::Lt(indvar, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // a[i] = i
    auto& block = builder.add_block(map_loop.root());
    auto& i_access = builder.add_access(block, "i");
    auto& a_access = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, i_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_access, {indvar});

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    ASSERT_EQ(root.size(), 1);
    EXPECT_EQ(&root.at(0), &outer_loop);
    ASSERT_EQ(outer_loop.root().size(), 1);
    EXPECT_EQ(&outer_loop.root().at(0), &block);
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

TEST(DeadCFGEliminationTest, TrivialMapWithSymbolicInit) {
    // for (j = 0; j < N; j++) { map (i = N; i < N + 1; i++) { for (k = 0; k < N; k++) {} } }
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    builder.add_container("k", int_type);

    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");

    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    auto indvar = symbolic::symbol("i");
    auto init = N;
    auto bound = symbolic::add(N, symbolic::integer(1));

    auto& map_loop = builder.add_map(
        outer_loop.root(),
        indvar,
        symbolic::Lt(indvar, bound),
        init,
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto k = symbolic::symbol("k");
    auto& inner_loop =
        builder
            .add_for(map_loop.root(), k, symbolic::Lt(k, N), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    ASSERT_EQ(root.size(), 1);
    ASSERT_EQ(outer_loop.root().size(), 1);
    EXPECT_EQ(&outer_loop.root().at(0), &inner_loop);
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

TEST(DeadCFGEliminationTest, TrivialMapWithLessThanOrEqual) {
    // for (j = 0; j < N; j++) { map (i = 0; i <= 0; i++) { for (k = 0; k < N; k++) {} } }
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    builder.add_container("k", int_type);

    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");

    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    // Executes once (bound becomes 0+1=1, trip count = 1-0=1)
    auto indvar = symbolic::symbol("i");
    auto& map_loop = builder.add_map(
        outer_loop.root(),
        indvar,
        symbolic::Le(indvar, symbolic::integer(0)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto k = symbolic::symbol("k");
    auto& inner_loop =
        builder
            .add_for(map_loop.root(), k, symbolic::Lt(k, N), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    ASSERT_EQ(root.size(), 1);
    ASSERT_EQ(outer_loop.root().size(), 1);
    EXPECT_EQ(&outer_loop.root().at(0), &inner_loop);
}

TEST(DeadCFGEliminationTest, TrivialMapEmptyBody) {
    // for (j = 0; j < N; j++) { map (i = 0; i < 1; i++) {} }
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);

    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");

    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    auto indvar = symbolic::symbol("i");
    auto& map_loop = builder.add_map(
        outer_loop.root(),
        indvar,
        symbolic::Lt(indvar, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));

    ASSERT_EQ(root.size(), 1);
    EXPECT_EQ(&root.at(0), &outer_loop);
    EXPECT_EQ(outer_loop.root().size(), 0);
}

TEST(DeadCFGEliminationTest, TrivialMapUseOfIndvar) {
    // for (j = 0; j < N; j++) { map (i = 0; i < 1; i++) { for (k = 0; k < N; k++) { a[i] = i; } } }
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar int_type(types::PrimitiveType::Int64);
    types::Pointer int_pointer_type(int_type);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    builder.add_container("j", int_type);
    builder.add_container("k", int_type);
    builder.add_container("a", int_pointer_type);

    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");

    auto j = symbolic::symbol("j");
    auto& outer_loop =
        builder.add_for(root, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));

    auto indvar = symbolic::symbol("i");
    auto& map = builder.add_map(
        outer_loop.root(),
        indvar,
        symbolic::Lt(indvar, symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto k = symbolic::symbol("k");
    auto& inner_loop =
        builder.add_for(map.root(), k, symbolic::Lt(k, N), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    // a[i] = i
    auto& block = builder.add_block(inner_loop.root());
    auto& i_access = builder.add_access(block, "i");
    auto& a_access = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, i_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_access, {indvar});

    EXPECT_EQ(outer_loop.root().size(), 1);
    EXPECT_EQ(map.root().size(), 1);
    dump_sdfg(builder.subject(), "0.before");

    // Dead CFG Elimination
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::DeadCFGElimination dce_pass;
    EXPECT_TRUE(dce_pass.run(builder, analysis_manager));
    dump_sdfg(builder.subject(), "1.after");

    ASSERT_EQ(root.size(), 1);
    ASSERT_EQ(outer_loop.root().size(), 1);
    ASSERT_EQ(&outer_loop.root().at(0), &inner_loop);
    ASSERT_EQ(inner_loop.root().size(), 1);
    auto* new_block = dyn_cast<structured_control_flow::Block*>(&inner_loop.root().at(0));
    ASSERT_NE(new_block, nullptr);
    auto data_nodes = new_block->dataflow().data_nodes();
    ASSERT_EQ(data_nodes.size(), 2);
    auto it = data_nodes.begin();
    auto* first_node = *it;
    it++;
    auto* second_node = *it;

    data_flow::ConstantNode* constant_node = nullptr;
    data_flow::AccessNode* access_node = nullptr;
    if (is_a(first_node->type_id(), ElementType::ConstantNode)) {
        constant_node = dyn_cast<data_flow::ConstantNode*>(first_node);
        access_node = dyn_cast<data_flow::AccessNode*>(second_node);
    } else {
        constant_node = dyn_cast<data_flow::ConstantNode*>(second_node);
        access_node = dyn_cast<data_flow::AccessNode*>(first_node);
    }

    ASSERT_NE(constant_node, nullptr);
    EXPECT_EQ(constant_node->data(), "0");
    ASSERT_NE(access_node, nullptr);
    EXPECT_EQ(access_node->data(), "a");
}
