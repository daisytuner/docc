#include "sdfg/transformations/loop_distribute.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

TEST(LoopDistributeTest, BasicDistribution) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add first computation (will be distributed)
    auto& block1 = builder.add_block(body);
    auto& A_in = builder.add_access(block1, "A");
    auto& A_out = builder.add_access(block1, "A");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, A_in, tasklet1, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", A_out, {symbolic::symbol("i")}, desc);

    // Add second computation
    auto& block2 = builder.add_block(body);
    auto& B_in = builder.add_access(block2, "B");
    auto& B_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, B_in, tasklet2, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", B_out, {symbolic::symbol("i")}, desc);

    // Verify initial state
    EXPECT_EQ(body.size(), 2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopDistribute transformation(loop);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));

    // Store element counts before transformation
    size_t initial_loop_count = 1;

    transformation.apply(builder, analysis_manager);

    // Verify transformation created a new loop
    auto& new_sdfg = builder.subject();
    EXPECT_EQ(new_sdfg.root().size(), 2); // Now we should have 2 loops

    // First loop should contain the first block
    auto first_loop = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(0).first);
    EXPECT_TRUE(first_loop != nullptr);
    EXPECT_EQ(first_loop->root().size(), 1);

    // Second loop should contain the second block
    auto second_loop = dynamic_cast<structured_control_flow::For*>(&new_sdfg.root().at(1).first);
    EXPECT_TRUE(second_loop != nullptr);
    EXPECT_EQ(second_loop->root().size(), 1);
    EXPECT_EQ(&second_loop->root().at(0).first, &block2);
}

TEST(LoopDistributeTest, Serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add two blocks
    auto& block1 = builder.add_block(body);
    auto& A_in = builder.add_access(block1, "A");
    auto& A_out = builder.add_access(block1, "A");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, A_in, tasklet1, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", A_out, {symbolic::symbol("i")}, desc);

    auto& block2 = builder.add_block(body);
    auto& B_in = builder.add_access(block2, "B");
    auto& B_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, B_in, tasklet2, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", B_out, {symbolic::symbol("i")}, desc);

    size_t loop_id = loop.element_id();

    transformations::LoopDistribute transformation(loop);

    // Test to_json
    nlohmann::json j;
    EXPECT_NO_THROW(transformation.to_json(j));

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "LoopDistribute");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_TRUE(j["subgraph"].contains("0"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], loop_id);
    EXPECT_TRUE(j["subgraph"]["0"]["type"] == "for" || j["subgraph"]["0"]["type"] == "map");
}

TEST(LoopDistributeTest, Deserialization) {
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

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")}, desc);

    size_t loop_id = loop.element_id();

    // Create JSON description
    nlohmann::json j;
    j["transformation_type"] = "LoopDistribute";
    j["subgraph"] = {{"0", {{"element_id", loop_id}, {"type", "for"}}}};

    // Test from_json
    EXPECT_NO_THROW({
        auto deserialized = transformations::LoopDistribute::from_json(builder, j);
        EXPECT_EQ(deserialized.name(), "LoopDistribute");
    });
}

TEST(LoopDistributeTest, CannotApplySingleChild) {
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

    // Define loop with only one child
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop = builder.add_for(root, indvar, condition, init, update);
    auto& body = loop.root();

    // Add only one computation
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {symbolic::symbol("i")}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {symbolic::symbol("i")}, desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::LoopDistribute transformation(loop);

    // Should not be applicable with only one child
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}
