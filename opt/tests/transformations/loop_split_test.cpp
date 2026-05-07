#include "sdfg/transformations/loop_split.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"

using namespace sdfg;

/// Helper: build a simple SDFG with a single loop for(i = 0; i < N; i++) { A[i] = A[i] }
static builder::StructuredSDFGBuilder make_simple_loop_sdfg(structured_control_flow::For*& out_loop) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")}, desc);

    out_loop = &loop;
    return builder;
}

/// Helper: cleanup passes (SequenceFusion + DeadCFGElimination)
static void cleanup(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& am) {
    bool applies = false;
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);
}

TEST(LoopSplitTest, Basic) {
    structured_control_flow::For* orig_loop = nullptr;
    auto builder = make_simple_loop_sdfg(orig_loop);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Split at symbolic point M (need to add M as a container)
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder_opt.add_container("M", sym_desc, true);
    auto split_point = symbolic::symbol("M");

    transformations::LoopSplit transformation(*orig_loop, split_point);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    cleanup(builder_opt, analysis_manager);

    auto& sdfg_opt = builder_opt.subject();
    // Should have two loops at root level
    EXPECT_EQ(sdfg_opt.root().size(), 2);

    // First loop: for(i_0 = 0; i_0 < M && i_0 < N; i_0++)
    auto* first_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first);
    ASSERT_TRUE(first_loop != nullptr);
    EXPECT_TRUE(symbolic::eq(first_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::
                    eq(first_loop->condition(),
                       symbolic::
                           And(symbolic::Lt(first_loop->indvar(), split_point),
                               symbolic::Lt(first_loop->indvar(), symbolic::symbol("N")))));
    EXPECT_EQ(first_loop->root().size(), 1);

    // Second loop: for(i = M; i < N; i++)
    auto* second_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(1).first);
    ASSERT_TRUE(second_loop != nullptr);
    EXPECT_EQ(second_loop->indvar()->get_name(), "i");
    EXPECT_TRUE(symbolic::eq(second_loop->init(), split_point));
    auto bound = symbolic::symbol("N");
    EXPECT_TRUE(symbolic::eq(second_loop->condition(), symbolic::Lt(second_loop->indvar(), bound)));
    EXPECT_EQ(second_loop->root().size(), 1);

    // Both should have body with a block
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&first_loop->root().at(0).first) != nullptr);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&second_loop->root().at(0).first) != nullptr);

    // First loop should have a fresh indvar
    EXPECT_NE(first_loop->indvar()->get_name(), "i");
}

TEST(LoopSplitTest, SplitAtConstant) {
    structured_control_flow::For* orig_loop = nullptr;
    auto builder = make_simple_loop_sdfg(orig_loop);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Split at constant 42
    auto split_point = symbolic::integer(42);

    transformations::LoopSplit transformation(*orig_loop, split_point);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    cleanup(builder_opt, analysis_manager);

    auto& sdfg_opt = builder_opt.subject();
    EXPECT_EQ(sdfg_opt.root().size(), 2);

    // First loop: i_0 in [0, 42) intersected with original bound (i_0 < N)
    auto* first_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first);
    ASSERT_TRUE(first_loop != nullptr);
    EXPECT_TRUE(symbolic::eq(first_loop->init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::
                    eq(first_loop->condition(),
                       symbolic::
                           And(symbolic::Lt(first_loop->indvar(), symbolic::integer(42)),
                               symbolic::Lt(first_loop->indvar(), symbolic::symbol("N")))));

    // Second loop: i in [42, N)
    auto* second_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(1).first);
    ASSERT_TRUE(second_loop != nullptr);
    EXPECT_TRUE(symbolic::eq(second_loop->init(), symbolic::integer(42)));
}

TEST(LoopSplitTest, SplitWithNonZeroInit) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Loop from M to N
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::symbol("M");
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")}, desc);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto split_point = symbolic::symbol("K");
    transformations::LoopSplit transformation(loop, split_point);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    cleanup(builder_opt, analysis_manager);

    auto& sdfg_opt = builder_opt.subject();
    EXPECT_EQ(sdfg_opt.root().size(), 2);

    // First loop: [M, K) intersected with original bound (i_0 < N)
    auto* first_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first);
    ASSERT_TRUE(first_loop != nullptr);
    EXPECT_TRUE(symbolic::eq(first_loop->init(), init));
    EXPECT_TRUE(symbolic::eq(
        first_loop->condition(),
        symbolic::And(symbolic::Lt(first_loop->indvar(), split_point), symbolic::Lt(first_loop->indvar(), bound))
    ));

    // Second loop: [K, N)
    auto* second_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(1).first);
    ASSERT_TRUE(second_loop != nullptr);
    EXPECT_TRUE(symbolic::eq(second_loop->init(), split_point));
    EXPECT_TRUE(symbolic::eq(second_loop->condition(), symbolic::Lt(second_loop->indvar(), bound)));
}

TEST(LoopSplitTest, Serialization) {
    structured_control_flow::For* orig_loop = nullptr;
    auto builder = make_simple_loop_sdfg(orig_loop);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("M", sym_desc, true);

    auto split_point = symbolic::symbol("M");
    size_t loop_id = orig_loop->element_id();

    transformations::LoopSplit transformation(*orig_loop, split_point);

    nlohmann::json j;
    EXPECT_NO_THROW(transformation.to_json(j));

    EXPECT_EQ(j["transformation_type"], "LoopSplit");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_TRUE(j.contains("parameters"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], loop_id);
    EXPECT_EQ(j["subgraph"]["0"]["type"], "for");
    EXPECT_EQ(j["parameters"]["split_point"], "M");
}

TEST(LoopSplitTest, Deserialization) {
    structured_control_flow::For* orig_loop = nullptr;
    auto builder = make_simple_loop_sdfg(orig_loop);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("M", sym_desc, true);

    size_t loop_id = orig_loop->element_id();

    nlohmann::json j;
    j["transformation_type"] = "LoopSplit";
    j["subgraph"] = {{"0", {{"element_id", loop_id}, {"type", "for"}}}};
    j["parameters"] = {{"split_point", "M"}};

    EXPECT_NO_THROW({
        auto deserialized = transformations::LoopSplit::from_json(builder, j);
        EXPECT_EQ(deserialized.name(), "LoopSplit");
    });
}

TEST(LoopSplitTest, CannotApplyNonContiguous) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    // Stride of 2 — not contiguous
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& loop = builder.add_for(root, indvar, condition, init, update);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    auto split_point = symbolic::integer(5);
    transformations::LoopSplit transformation(loop, split_point);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(LoopSplitTest, SplitExpressionArithmetic) {
    structured_control_flow::For* orig_loop = nullptr;
    auto builder = make_simple_loop_sdfg(orig_loop);

    auto structured_sdfg = builder.move();
    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Split at N/2 (arithmetic expression as split point)
    auto split_point = symbolic::div(symbolic::symbol("N"), symbolic::integer(2));

    transformations::LoopSplit transformation(*orig_loop, split_point);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    cleanup(builder_opt, analysis_manager);

    auto& sdfg_opt = builder_opt.subject();
    EXPECT_EQ(sdfg_opt.root().size(), 2);

    // First loop: [0, N/2)
    auto* first_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first);
    ASSERT_TRUE(first_loop != nullptr);
    EXPECT_TRUE(symbolic::eq(first_loop->init(), symbolic::integer(0)));

    // Second loop: [N/2, N)
    auto* second_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(1).first);
    ASSERT_TRUE(second_loop != nullptr);
    EXPECT_TRUE(symbolic::eq(second_loop->init(), split_point));
}
