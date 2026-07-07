#include "sdfg/transformations/multi_level_tiling.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"

using namespace sdfg;

TEST(MultiLevelTilingTest, TwoLevelTiling) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop: for i = 0; i < N; i++
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& orig_loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = orig_loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")}, desc);

    auto structured_sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(structured_sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    // Apply two-level tiling: outer tile = 64, inner tile = 8
    // Expected result: i_tile (step 64) -> i_tile1 (step 8) -> i (step 1)
    transformations::MultiLevelTiling transformation(orig_loop, 64, 8);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    // Cleanup
    bool applies = false;
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    do {
        applies = false;
        applies |= dead_cfg.run(builder_opt, analysis_manager);
        applies |= sequence_fusion.run(builder_opt, analysis_manager);
    } while (applies);

    auto& sdfg_opt = builder_opt.subject();

    // Should have 3 nested loops: outer tile -> middle tile -> inner point
    EXPECT_EQ(sdfg_opt.root().size(), 1);
    auto* outer_loop = dyn_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first);
    ASSERT_NE(outer_loop, nullptr);
    EXPECT_EQ(outer_loop->indvar()->get_name(), "i_tile0");

    // Outer loop steps by 64
    EXPECT_TRUE(symbolic::eq(outer_loop->update(), symbolic::add(outer_loop->indvar(), symbolic::integer(64))));

    // Middle loop
    EXPECT_EQ(outer_loop->root().size(), 1);
    auto* middle_loop = dyn_cast<structured_control_flow::For*>(&outer_loop->root().at(0).first);
    ASSERT_NE(middle_loop, nullptr);

    // Middle loop init is the outer indvar
    EXPECT_TRUE(symbolic::eq(middle_loop->init(), outer_loop->indvar()));

    // Middle loop steps by 8
    EXPECT_TRUE(symbolic::eq(middle_loop->update(), symbolic::add(middle_loop->indvar(), symbolic::integer(8))));

    // Inner (point) loop
    EXPECT_EQ(middle_loop->root().size(), 1);
    auto* inner_loop = dyn_cast<structured_control_flow::For*>(&middle_loop->root().at(0).first);
    ASSERT_NE(inner_loop, nullptr);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");

    // Inner loop init is the middle indvar
    EXPECT_TRUE(symbolic::eq(inner_loop->init(), middle_loop->indvar()));

    // Inner loop steps by 1
    EXPECT_TRUE(symbolic::eq(inner_loop->update(), symbolic::add(inner_loop->indvar(), symbolic::integer(1))));

    // Block is preserved in the innermost loop
    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Block*>(&inner_loop->root().at(0).first) != nullptr);

    // Both tile variables exist
    EXPECT_TRUE(sdfg_opt.exists("i_tile0"));
    EXPECT_TRUE(sdfg_opt.exists(middle_loop->indvar()->get_name()));

    // Verify accessor methods
    EXPECT_EQ(transformation.outer_loop(), outer_loop);
    EXPECT_EQ(transformation.middle_loop(), middle_loop);
    EXPECT_EQ(transformation.inner_loop(), inner_loop);
}

TEST(MultiLevelTilingTest, TwoLevelTilingSerialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& orig_loop = builder.add_for(root, indvar, condition, init, update);

    size_t loop_id = orig_loop.element_id();

    // Serialize
    transformations::MultiLevelTiling transformation(orig_loop, 64, 8);
    nlohmann::json j;
    EXPECT_NO_THROW(transformation.to_json(j));

    EXPECT_EQ(j["transformation_type"], "MultiLevelTiling");
    EXPECT_EQ(j["parameters"]["tile_size"], 64);
    EXPECT_EQ(j["parameters"]["tile_size_2"], 8);

    // Deserialize
    auto deserialized = transformations::MultiLevelTiling::from_json(builder, j);
    EXPECT_EQ(deserialized.name(), "MultiLevelTiling");

    // Verify it can be applied
    analysis::AnalysisManager am(builder.subject());
    EXPECT_TRUE(deserialized.can_be_applied(builder, am));
}

TEST(MultiLevelTilingTest, TwoLevelTilingInvalidParameters) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto condition = symbolic::Lt(indvar, symbolic::symbol("N"));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& orig_loop = builder.add_for(root, indvar, condition, init, update);

    analysis::AnalysisManager am(builder.subject());

    // tile_size_2 == 1 should be rejected
    transformations::MultiLevelTiling t1(orig_loop, 64, 1);
    EXPECT_FALSE(t1.can_be_applied(builder, am));

    // tile_size_2 >= tile_size should be rejected
    transformations::MultiLevelTiling t2(orig_loop, 64, 64);
    EXPECT_FALSE(t2.can_be_applied(builder, am));

    transformations::MultiLevelTiling t3(orig_loop, 64, 128);
    EXPECT_FALSE(t3.can_be_applied(builder, am));
    // tile_size not divisible by tile_size_2 should be rejected
    transformations::MultiLevelTiling t4(orig_loop, 64, 6);
    EXPECT_FALSE(t4.can_be_applied(builder, am));

    transformations::MultiLevelTiling t5(orig_loop, 64, 5);
    EXPECT_FALSE(t5.can_be_applied(builder, am));

    // tile_size divisible by tile_size_2 should be accepted
    transformations::MultiLevelTiling t6(orig_loop, 64, 8);
    EXPECT_TRUE(t6.can_be_applied(builder, am));

    transformations::MultiLevelTiling t7(orig_loop, 64, 32);
    EXPECT_TRUE(t7.can_be_applied(builder, am));
}
