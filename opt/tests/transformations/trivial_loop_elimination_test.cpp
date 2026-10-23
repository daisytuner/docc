#include "sdfg/transformations/trivial_loop_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/passes/trivial_loop_elimination_pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace {

structured_control_flow::Map& add_seq_map(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::string& indvar_name,
    symbolic::Condition condition,
    symbolic::Expression init
) {
    auto indvar = symbolic::symbol(indvar_name);
    return builder.add_map(
        parent,
        indvar,
        condition,
        init,
        symbolic::add(indvar, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
}

// a[i] = i inside `map`
void add_store_indvar(builder::StructuredSDFGBuilder& builder, structured_control_flow::Map& map) {
    auto& block = builder.add_block(map.root());
    auto& i_access = builder.add_access(block, map.indvar()->get_name());
    auto& a_access = builder.add_access(block, "a");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, i_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", a_access, {map.indvar()});
}

} // namespace

TEST(TrivialLoopEliminationTest, TrivialMap) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("i", int_type);
    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& map = add_seq_map(builder, root, "i", symbolic::Lt(i, symbolic::integer(1)), symbolic::zero());
    builder.add_block(map.root());

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TrivialLoopElimination transformation(map);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    ASSERT_EQ(root.size(), 1);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&root.at(0)), nullptr);
}

TEST(TrivialLoopEliminationTest, TrivialMapEmptyBody) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("i", int_type);
    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& map = add_seq_map(builder, root, "i", symbolic::Lt(i, symbolic::integer(1)), symbolic::zero());

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TrivialLoopElimination transformation(map);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    EXPECT_EQ(root.size(), 0);
}

TEST(TrivialLoopEliminationTest, TrivialMapWithLessThanOrEqual) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("i", int_type);
    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& map = add_seq_map(builder, root, "i", symbolic::Le(i, symbolic::zero()), symbolic::zero());
    builder.add_block(map.root());

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TrivialLoopElimination transformation(map);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    ASSERT_EQ(root.size(), 1);
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&root.at(0)), nullptr);
}

TEST(TrivialLoopEliminationTest, NonTrivialMap) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("i", int_type);
    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& map = add_seq_map(builder, root, "i", symbolic::Lt(i, symbolic::integer(10)), symbolic::zero());
    builder.add_block(map.root());

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TrivialLoopElimination transformation(map);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(TrivialLoopEliminationTest, NonSequentialMap) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("i", int_type);
    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& map = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(1)),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        structured_control_flow::ScheduleType("TEST_OFFLOAD", structured_control_flow::ScheduleTypeCategory::Offloader)
    );
    builder.add_block(map.root());

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TrivialLoopElimination transformation(map);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(TrivialLoopEliminationTest, UseOfIndvarIntegerInit) {
    // map (i = 3; i < 4; i++) { a[i] = i; }  ->  a[3] = 3
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    types::Pointer int_pointer_type(int_type);
    builder.add_container("i", int_type);
    builder.add_container("a", int_pointer_type);
    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& map = add_seq_map(builder, root, "i", symbolic::Lt(i, symbolic::integer(4)), symbolic::integer(3));
    add_store_indvar(builder, map);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TrivialLoopElimination transformation(map);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    ASSERT_EQ(root.size(), 1);
    auto* block = dyn_cast<structured_control_flow::Block*>(&root.at(0));
    ASSERT_NE(block, nullptr);

    data_flow::ConstantNode* constant_node = nullptr;
    data_flow::AccessNode* a_node = nullptr;
    for (auto* node : block->dataflow().data_nodes()) {
        if (is_a(node->type_id(), ElementType::ConstantNode)) {
            constant_node = dyn_cast<data_flow::ConstantNode*>(node);
        } else {
            a_node = node;
        }
    }
    ASSERT_NE(constant_node, nullptr);
    EXPECT_EQ(constant_node->data(), "3");
    ASSERT_NE(a_node, nullptr);
    EXPECT_EQ(a_node->data(), "a");
    auto& memlet = *block->dataflow().in_edges(*a_node).begin();
    ASSERT_EQ(memlet.subset().size(), 1);
    EXPECT_TRUE(symbolic::eq(memlet.subset().at(0), symbolic::integer(3)));
}

TEST(TrivialLoopEliminationTest, UseOfIndvarSymbolicInit) {
    // map (i = N; i < N + 1; i++) { a[i] = i; }  ->  a[N] = N
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    types::Pointer int_pointer_type(int_type);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    builder.add_container("a", int_pointer_type);
    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    auto& map = add_seq_map(builder, root, "i", symbolic::Lt(i, symbolic::add(N, symbolic::one())), N);
    add_store_indvar(builder, map);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TrivialLoopElimination transformation(map);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    ASSERT_EQ(root.size(), 1);
    auto* block = dyn_cast<structured_control_flow::Block*>(&root.at(0));
    ASSERT_NE(block, nullptr);
    for (auto* node : block->dataflow().data_nodes()) {
        EXPECT_NE(node->data(), "i");
    }
}

TEST(TrivialLoopEliminationTest, UseOfIndvarExpressionInit) {
    // map (i = N + 2; i < N + 3; i++) { a[i] = i; }  ->  i = N + 2; a[N + 2] = i
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    types::Pointer int_pointer_type(int_type);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    builder.add_container("a", int_pointer_type);
    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    auto init = symbolic::add(N, symbolic::integer(2));
    auto& map = add_seq_map(builder, root, "i", symbolic::Lt(i, symbolic::add(N, symbolic::integer(3))), init);
    add_store_indvar(builder, map);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::TrivialLoopElimination transformation(map);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    ASSERT_EQ(root.size(), 2);
    auto* assignments = dyn_cast<structured_control_flow::AssignmentBlock*>(&root.at(0));
    ASSERT_NE(assignments, nullptr);
    ASSERT_EQ(assignments->assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(assignments->assignments().at(i), init));
    EXPECT_NE(dyn_cast<structured_control_flow::Block*>(&root.at(1)), nullptr);
}

TEST(TrivialLoopEliminationTest, Serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("i", int_type);
    auto& root = builder.subject().root();

    auto i = symbolic::symbol("i");
    auto& map = add_seq_map(builder, root, "i", symbolic::Lt(i, symbolic::integer(1)), symbolic::zero());

    transformations::TrivialLoopElimination transformation(map);
    nlohmann::json j;
    transformation.to_json(j);
    EXPECT_EQ(j["transformation_type"], "TrivialLoopElimination");

    auto restored = transformations::TrivialLoopElimination::from_json(builder, j);
    EXPECT_EQ(restored.name(), "TrivialLoopElimination");
}

TEST(TrivialLoopEliminationPassTest, PromotesNestedMaps) {
    // map (b = 0; b < 1) { map (i = 0; i < N) {...}; map (j = 0; j < 1) { map (k = 0; k < N) {...} } }
    // -> [map i, map k]
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("N", int_type, true);
    for (auto name : {"b", "i", "j", "k"}) {
        builder.add_container(name, int_type);
    }
    auto& root = builder.subject().root();
    auto N = symbolic::symbol("N");

    auto& outer =
        add_seq_map(builder, root, "b", symbolic::Lt(symbolic::symbol("b"), symbolic::one()), symbolic::zero());
    auto& map_i = add_seq_map(builder, outer.root(), "i", symbolic::Lt(symbolic::symbol("i"), N), symbolic::zero());
    builder.add_block(map_i.root());
    auto& map_j =
        add_seq_map(builder, outer.root(), "j", symbolic::Lt(symbolic::symbol("j"), symbolic::one()), symbolic::zero());
    auto& map_k = add_seq_map(builder, map_j.root(), "k", symbolic::Lt(symbolic::symbol("k"), N), symbolic::zero());
    builder.add_block(map_k.root());

    auto& other = add_seq_map(builder, root, "i", symbolic::Lt(symbolic::symbol("i"), N), symbolic::zero());
    builder.add_block(other.root());

    std::vector<structured_control_flow::Map*> maps = {&outer, &other};
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::TrivialLoopEliminationPass pass(maps);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    ASSERT_EQ(maps.size(), 3);
    EXPECT_EQ(maps[0], &map_i);
    EXPECT_EQ(maps[1], &map_k);
    EXPECT_EQ(maps[2], &other);

    ASSERT_EQ(root.size(), 3);
    EXPECT_EQ(&root.at(0), &map_i);
    EXPECT_EQ(&root.at(1), &map_k);
    EXPECT_EQ(&root.at(2), &other);
}

TEST(TrivialLoopEliminationPassTest, NotApplicable) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar int_type(types::PrimitiveType::Int64);
    builder.add_container("N", int_type, true);
    builder.add_container("i", int_type);
    auto& root = builder.subject().root();

    auto& map =
        add_seq_map(builder, root, "i", symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")), symbolic::zero());
    builder.add_block(map.root());

    std::vector<structured_control_flow::Map*> maps = {&map};
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::TrivialLoopEliminationPass pass(maps);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
    ASSERT_EQ(maps.size(), 1);
    EXPECT_EQ(maps[0], &map);
}
