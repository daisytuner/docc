#include "sdfg/transformations/loop_tiling.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"

using namespace sdfg;

TEST(LoopTilingTest, Basic) {
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

    // Define loop
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

    // Apply
    transformations::LoopTiling transformation(orig_loop, 32);
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
    EXPECT_EQ(sdfg_opt.root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first) != nullptr);
    auto& loop = static_cast<structured_control_flow::For&>(sdfg_opt.root().at(0).first);

    EXPECT_EQ(loop.root().size(), 1);

    // Check
    EXPECT_EQ(loop.indvar()->get_name(), "i_tile0");

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&loop.root().at(0).first) != nullptr);
    auto inner_loop = static_cast<structured_control_flow::For*>(&loop.root().at(0).first);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");

    auto& outer_update = loop.update();
    EXPECT_TRUE(symbolic::eq(outer_update, symbolic::add(loop.indvar(), symbolic::integer(32))));

    auto& inner_init = inner_loop->init();
    EXPECT_TRUE(symbolic::eq(inner_init, loop.indvar()));

    auto& inner_condition_tile = inner_loop->condition();
    EXPECT_TRUE(symbolic::
                    eq(inner_condition_tile,
                       symbolic::
                           And(symbolic::Lt(inner_loop->indvar(), symbolic::add(loop.indvar(), symbolic::integer(32))),
                               symbolic::Lt(inner_loop->indvar(), bound))));
    auto& inner_update = inner_loop->update();
    EXPECT_TRUE(symbolic::eq(inner_update, symbolic::add(inner_loop->indvar(), symbolic::integer(1))));

    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&inner_loop->root().at(0).first) != nullptr);

    EXPECT_EQ(builder_opt.subject().exists("i_tile0"), true);
}

TEST(LoopTilingTest, Basic_Transition) {
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

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& orig_loop = builder.add_for(root, indvar, condition, init, update, {{indvar, symbolic::zero()}});
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

    // Apply
    transformations::LoopTiling transformation(orig_loop, 32);
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
    EXPECT_EQ(sdfg_opt.root().size(), 1);
    EXPECT_EQ(sdfg_opt.root().at(0).second.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(sdfg_opt.root().at(0).second.assignments().at(indvar), symbolic::zero()));
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first) != nullptr);
    auto& loop = static_cast<structured_control_flow::For&>(sdfg_opt.root().at(0).first);

    EXPECT_EQ(loop.root().size(), 1);

    // Check
    EXPECT_EQ(loop.indvar()->get_name(), "i_tile0");

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&loop.root().at(0).first) != nullptr);
    EXPECT_EQ(loop.root().at(0).second.assignments().size(), 0);
    auto inner_loop = static_cast<structured_control_flow::For*>(&loop.root().at(0).first);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");

    auto& outer_update = loop.update();
    EXPECT_TRUE(symbolic::eq(outer_update, symbolic::add(loop.indvar(), symbolic::integer(32))));

    auto& inner_init = inner_loop->init();
    EXPECT_TRUE(symbolic::eq(inner_init, loop.indvar()));

    auto& inner_condition_tile = inner_loop->condition();
    EXPECT_TRUE(symbolic::
                    eq(inner_condition_tile,
                       symbolic::
                           And(symbolic::Lt(inner_loop->indvar(), symbolic::add(loop.indvar(), symbolic::integer(32))),
                               symbolic::Lt(inner_loop->indvar(), bound))));
    auto& inner_update = inner_loop->update();
    EXPECT_TRUE(symbolic::eq(inner_update, symbolic::add(inner_loop->indvar(), symbolic::integer(1))));

    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Block*>(&inner_loop->root().at(0).first) != nullptr);

    EXPECT_EQ(builder_opt.subject().exists("i_tile0"), true);
}

TEST(LoopTilingTest, Serialization) {
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

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& orig_loop = builder.add_for(root, indvar, condition, init, update);

    size_t loop_id = orig_loop.element_id();
    size_t tile_size = 32;

    transformations::LoopTiling transformation(orig_loop, tile_size);

    // Test to_json
    nlohmann::json j;
    EXPECT_NO_THROW(transformation.to_json(j));

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "LoopTiling");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_TRUE(j.contains("parameters"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], loop_id);
    EXPECT_EQ(j["subgraph"]["0"]["type"], "for");
    EXPECT_EQ(j["parameters"]["tile_size"], tile_size);
}

TEST(LoopTilingTest, Deserialization) {
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

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& orig_loop = builder.add_for(root, indvar, condition, init, update);

    size_t loop_id = orig_loop.element_id();

    // Create JSON description
    nlohmann::json j;
    j["transformation_type"] = "LoopTiling";
    j["subgraph"] = {{"0", {{"element_id", loop_id}, {"type", "for"}}}};
    j["parameters"] = {{"tile_size", 16}};

    // Test from_json
    EXPECT_NO_THROW({
        auto deserialized = transformations::LoopTiling::from_json(builder, j);
        EXPECT_EQ(deserialized.name(), "LoopTiling");
    });
}

TEST(LoopTilingTest, CreatedElements) {
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

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& orig_loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = orig_loop.root();

    // Add computation
    auto& block = builder.add_block(body);

    // Store original state
    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(body.size(), 1);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopTiling transformation(orig_loop, 32);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // After tiling, we should have created a new outer loop and a new induction variable
    auto& sdfg_opt = builder.subject();
    EXPECT_EQ(sdfg_opt.root().size(), 1);

    // Verify outer loop was created
    auto outer_loop = dynamic_cast<structured_control_flow::For*>(&sdfg_opt.root().at(0).first);
    EXPECT_TRUE(outer_loop != nullptr);

    // Verify new tile induction variable was created
    std::string tile_var_name = "i_tile0";
    EXPECT_TRUE(sdfg_opt.exists(tile_var_name));
    EXPECT_EQ(outer_loop->indvar()->get_name(), tile_var_name);

    // Verify inner loop exists and has the original induction variable
    EXPECT_EQ(outer_loop->root().size(), 1);
    auto inner_loop = dynamic_cast<structured_control_flow::For*>(&outer_loop->root().at(0).first);
    EXPECT_TRUE(inner_loop != nullptr);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");

    // Verify block is preserved in the inner loop
    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_EQ(&inner_loop->root().at(0).first, &block);
}
