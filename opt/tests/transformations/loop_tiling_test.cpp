#include "sdfg/transformations/loop_tiling.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"

using namespace sdfg;

TEST(LoopTilingTest, For_Integer) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::integer(64);
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

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopTiling transformation(orig_loop, 4);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    EXPECT_THROW(transformation.inner_loop(), InvalidSDFGException);
    EXPECT_THROW(transformation.outer_loop(), InvalidSDFGException);
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(transformation.inner_loop());
    EXPECT_NO_THROW(transformation.outer_loop());

    EXPECT_EQ(sdfg.root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&sdfg.root().at(0).first) != nullptr);
    auto& loop = static_cast<structured_control_flow::For&>(sdfg.root().at(0).first);

    EXPECT_EQ(loop.root().size(), 1);
    EXPECT_EQ(loop.indvar()->get_name(), "i_tile0");
    EXPECT_EQ(builder.subject().exists("i_tile0"), true);
    EXPECT_TRUE(symbolic::eq(loop.update(), symbolic::add(loop.indvar(), symbolic::integer(4))));

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&loop.root().at(0).first) != nullptr);
    auto inner_loop = static_cast<structured_control_flow::For*>(&loop.root().at(0).first);
    EXPECT_EQ(inner_loop, &orig_loop);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");
    EXPECT_TRUE(symbolic::eq(inner_loop->init(), loop.indvar()));
    EXPECT_TRUE(symbolic::
                    eq(inner_loop->condition(),
                       symbolic::
                           And(symbolic::Lt(inner_loop->indvar(), symbolic::add(loop.indvar(), symbolic::integer(4))),
                               symbolic::Lt(inner_loop->indvar(), bound))));
    EXPECT_TRUE(symbolic::eq(inner_loop->update(), symbolic::add(inner_loop->indvar(), symbolic::integer(1))));

    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_TRUE(&inner_loop->root().at(0).first == &block);
}

TEST(LoopTilingTest, For_Symbolic) {
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

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopTiling transformation(orig_loop, 32);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    EXPECT_THROW(transformation.inner_loop(), InvalidSDFGException);
    EXPECT_THROW(transformation.outer_loop(), InvalidSDFGException);
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(transformation.inner_loop());
    EXPECT_NO_THROW(transformation.outer_loop());

    EXPECT_EQ(sdfg.root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&sdfg.root().at(0).first) != nullptr);
    auto& loop = static_cast<structured_control_flow::For&>(sdfg.root().at(0).first);

    EXPECT_EQ(loop.root().size(), 1);
    EXPECT_EQ(loop.indvar()->get_name(), "i_tile0");
    EXPECT_EQ(builder.subject().exists("i_tile0"), true);
    EXPECT_TRUE(symbolic::eq(loop.update(), symbolic::add(loop.indvar(), symbolic::integer(32))));

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&loop.root().at(0).first) != nullptr);
    auto inner_loop = static_cast<structured_control_flow::For*>(&loop.root().at(0).first);
    EXPECT_EQ(inner_loop, &orig_loop);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");
    EXPECT_TRUE(symbolic::eq(inner_loop->init(), loop.indvar()));
    EXPECT_TRUE(symbolic::
                    eq(inner_loop->condition(),
                       symbolic::
                           And(symbolic::Lt(inner_loop->indvar(), symbolic::add(loop.indvar(), symbolic::integer(32))),
                               symbolic::Lt(inner_loop->indvar(), bound))));
    EXPECT_TRUE(symbolic::eq(inner_loop->update(), symbolic::add(inner_loop->indvar(), symbolic::integer(1))));

    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_TRUE(&inner_loop->root().at(0).first == &block);
}

TEST(LoopTilingTest, For_WithTransition) {
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

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopTiling transformation(orig_loop, 32);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    EXPECT_THROW(transformation.inner_loop(), InvalidSDFGException);
    EXPECT_THROW(transformation.outer_loop(), InvalidSDFGException);
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(transformation.inner_loop());
    EXPECT_NO_THROW(transformation.outer_loop());

    EXPECT_EQ(sdfg.root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&sdfg.root().at(0).first) != nullptr);
    auto& loop = static_cast<structured_control_flow::For&>(sdfg.root().at(0).first);
    EXPECT_EQ(sdfg.root().at(0).second.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(sdfg.root().at(0).second.assignments().at(indvar), symbolic::zero()));

    EXPECT_EQ(loop.root().size(), 1);
    EXPECT_EQ(loop.indvar()->get_name(), "i_tile0");
    EXPECT_EQ(builder.subject().exists("i_tile0"), true);
    EXPECT_TRUE(symbolic::eq(loop.update(), symbolic::add(loop.indvar(), symbolic::integer(32))));

    EXPECT_TRUE(dynamic_cast<structured_control_flow::For*>(&loop.root().at(0).first) != nullptr);
    auto inner_loop = static_cast<structured_control_flow::For*>(&loop.root().at(0).first);
    EXPECT_EQ(inner_loop, &orig_loop);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");
    EXPECT_TRUE(symbolic::eq(inner_loop->init(), loop.indvar()));
    EXPECT_TRUE(symbolic::
                    eq(inner_loop->condition(),
                       symbolic::
                           And(symbolic::Lt(inner_loop->indvar(), symbolic::add(loop.indvar(), symbolic::integer(32))),
                               symbolic::Lt(inner_loop->indvar(), bound))));
    EXPECT_TRUE(symbolic::eq(inner_loop->update(), symbolic::add(inner_loop->indvar(), symbolic::integer(1))));

    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_TRUE(&inner_loop->root().at(0).first == &block);
}

TEST(LoopTilingTest, Map_Symbolic) {
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

    auto& orig_loop = builder.add_map(
        root,
        indvar,
        condition,
        init,
        update,
        structured_control_flow::ScheduleType_Sequential::create(),
        {{indvar, symbolic::zero()}}
    );
    auto& body = orig_loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopTiling transformation(orig_loop, 32);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    EXPECT_THROW(transformation.inner_loop(), InvalidSDFGException);
    EXPECT_THROW(transformation.outer_loop(), InvalidSDFGException);
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(transformation.inner_loop());
    EXPECT_NO_THROW(transformation.outer_loop());

    EXPECT_EQ(sdfg.root().size(), 1);
    EXPECT_TRUE(dynamic_cast<structured_control_flow::Map*>(&sdfg.root().at(0).first) != nullptr);
    auto& loop = static_cast<structured_control_flow::Map&>(sdfg.root().at(0).first);
    EXPECT_EQ(sdfg.root().at(0).second.assignments().size(), 1);
    EXPECT_TRUE(symbolic::eq(sdfg.root().at(0).second.assignments().at(indvar), symbolic::zero()));

    EXPECT_EQ(loop.root().size(), 1);
    EXPECT_EQ(loop.indvar()->get_name(), "i_tile0");
    EXPECT_EQ(builder.subject().exists("i_tile0"), true);
    EXPECT_TRUE(symbolic::eq(loop.update(), symbolic::add(loop.indvar(), symbolic::integer(32))));

    EXPECT_TRUE(dynamic_cast<structured_control_flow::Map*>(&loop.root().at(0).first) != nullptr);
    auto inner_loop = static_cast<structured_control_flow::Map*>(&loop.root().at(0).first);
    EXPECT_EQ(inner_loop, &orig_loop);
    EXPECT_EQ(inner_loop->indvar()->get_name(), "i");
    EXPECT_TRUE(symbolic::eq(inner_loop->init(), loop.indvar()));
    EXPECT_TRUE(symbolic::
                    eq(inner_loop->condition(),
                       symbolic::
                           And(symbolic::Lt(inner_loop->indvar(), symbolic::add(loop.indvar(), symbolic::integer(32))),
                               symbolic::Lt(inner_loop->indvar(), bound))));
    EXPECT_TRUE(symbolic::eq(inner_loop->update(), symbolic::add(inner_loop->indvar(), symbolic::integer(1))));

    EXPECT_EQ(inner_loop->root().size(), 1);
    EXPECT_TRUE(&inner_loop->root().at(0).first == &block);
}

TEST(LoopTilingTest, InvalidTileSize) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::integer(64);
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

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopTiling transformation(orig_loop, 0);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(LoopTilingTest, NonContiguous) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::integer(64);
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(2));

    auto& orig_loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = orig_loop.root();

    // Add computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopTiling transformation(orig_loop, 32);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
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

    EXPECT_THROW(
        {
            nlohmann::json invalid_j = j;
            invalid_j["subgraph"]["0"]["element_id"] = 9999; // Non-existent element ID
            transformations::LoopTiling::from_json(builder, invalid_j);
        },
        transformations::InvalidTransformationDescriptionException
    );
}
