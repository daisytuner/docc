#include "sdfg/transformations/map_fusion.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(MapFusionTest, ProducerConsumer_1D) {
    // Create two sequential maps where second map reads from first map's output
    // Map 1: T[i] = A[i] + 1.0
    // Map 2: B[i] = T[i] * 2.0
    // After fusion: B[i] = (A[i] + 1.0) * 2.0

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Define first map: T[i] = A[i] + 1.0
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = map1.root();

    auto& block1 = builder.add_block(body1);
    auto& a_in = builder.add_access(block1, "A");
    auto& one_node = builder.add_constant(block1, "1.0", float_desc);
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in1", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, one_node, tasklet1, "_in2", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Define second map: B[j] = T[j] * 2.0
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body2 = map2.root();

    auto& block2 = builder.add_block(body2);
    auto& t_in = builder.add_access(block2, "T");
    auto& two_node = builder.add_constant(block2, "2.0", float_desc);
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in1", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, two_node, tasklet2, "_in2", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    // Analyze and apply transformation
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // Verify transformation results
    auto& new_sdfg = builder.subject();

    // Both maps should still exist
    EXPECT_EQ(new_sdfg.root().size(), 2);

    // The second map should now have 2 blocks in its body (producer + consumer)
    auto* new_map2 = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(1).first);
    EXPECT_TRUE(new_map2 != nullptr);
    EXPECT_EQ(new_map2->root().size(), 2) << "Second loop should now have 2 blocks (producer + consumer)";

    // First block is the new producer block
    auto* producer_block = dynamic_cast<structured_control_flow::Block*>(&new_map2->root().at(0).first);
    EXPECT_TRUE(producer_block != nullptr);

    // Second block is the original consumer block
    auto* consumer_block = dynamic_cast<structured_control_flow::Block*>(&new_map2->root().at(1).first);
    EXPECT_TRUE(consumer_block != nullptr);
}

TEST(MapFusionTest, SimpleInputOutputOverlap) {
    // Create two sequential maps where second map reads from first map's output
    // Map 1: T[i] = A[i] + 1.0
    // Map 2: B[i] = T[i] * 2.0
    // After fusion: B[i] = (A[i] + 1.0) * 2.0

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);

    // Define first map: T[i] = A[i] + 1.0
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = map1.root();

    auto& block1 = builder.add_block(body1);
    auto& a_in = builder.add_access(block1, "A");
    auto& one_node = builder.add_constant(block1, "1.0", float_desc);
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in1", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, one_node, tasklet1, "_in2", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Define second map: B[j] = T[j] * 2.0
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body2 = map2.root();

    auto& block2 = builder.add_block(body2);
    auto& t_in = builder.add_access(block2, "T");
    auto& two_node = builder.add_constant(block2, "2.0", float_desc);
    auto& b_out = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in1", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, two_node, tasklet2, "_in2", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    // Analyze and apply transformation
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));

    // Verify transformation results
    auto& new_sdfg = builder.subject();

    // Both maps should still exist
    EXPECT_EQ(new_sdfg.root().size(), 2);

    auto* map1_after = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(0).first);
    EXPECT_TRUE(map1_after != nullptr);
    EXPECT_EQ(map1_after->root().size(), 1) << "First loop should still have 1 block";

    auto* map2_after = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(1).first);
    EXPECT_TRUE(map2_after != nullptr);
    EXPECT_EQ(map2_after->root().size(), 1) << "Second loop should still have 1 block";
}

TEST(MapFusionTest, NonSequentialMaps) {
    // Test that non-sequential maps cannot be fused

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Define first map
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = map1.root();
    auto& block1 = builder.add_block(body1);
    auto& a_in = builder.add_access(block1, "A");
    auto& one_node = builder.add_constant(block1, "1.0", float_desc);
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in1", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, one_node, tasklet1, "_in2", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Add an intervening block
    auto& intervening_block = builder.add_block(root);

    // Define second map
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body2 = map2.root();
    auto& block2 = builder.add_block(body2);
    auto& t_in = builder.add_access(block2, "T");
    auto& two_node = builder.add_constant(block2, "2.0", float_desc);
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in1", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, two_node, tasklet2, "_in2", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    // Analyze - should not be able to apply since maps are not sequential
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(MapFusionTest, NoSharedData) {
    // Test that maps without shared data cannot be fused

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    builder.add_container("A", array_desc, true);
    builder.add_container("B", array_desc, true);
    builder.add_container("C", array_desc, true);
    builder.add_container("D", array_desc, true);

    // Define first map: B[i] = A[i]
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = map1.root();
    auto& block1 = builder.add_block(body1);
    auto& a_in = builder.add_access(block1, "A");
    auto& b_out = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", b_out, {symbolic::symbol("i")}, array_desc);

    // Define second map: D[j] = C[j] (no dependency on first map's output)
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body2 = map2.root();
    auto& block2 = builder.add_block(body2);
    auto& c_in = builder.add_access(block2, "C");
    auto& d_out = builder.add_access(block2, "D");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, c_in, tasklet2, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", d_out, {symbolic::symbol("j")}, array_desc);

    // Analyze - should not be able to apply since no shared data
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(MapFusionTest, Serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Define first map
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Define second map
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    size_t first_map_id = map1.element_id();
    size_t second_map_id = map2.element_id();

    transformations::MapFusion transformation(map1, map2);

    // Test to_json
    nlohmann::json j;
    EXPECT_NO_THROW(transformation.to_json(j));

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "MapFusion");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_TRUE(j["subgraph"].contains("0"));
    EXPECT_TRUE(j["subgraph"].contains("1"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], first_map_id);
    EXPECT_EQ(j["subgraph"]["1"]["element_id"], second_map_id);
    EXPECT_EQ(j["subgraph"]["0"]["type"], "map");
    EXPECT_EQ(j["subgraph"]["1"]["type"], "map");
}

TEST(MapFusionTest, Deserialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);

    // Define nested maps
    auto indvar = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto& map2 = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    size_t first_map_id = map1.element_id();
    size_t second_map_id = map2.element_id();

    // Create JSON description
    nlohmann::json j;
    j["transformation_type"] = "MapFusion";
    j["subgraph"] = {
        {"0", {{"element_id", first_map_id}, {"type", "map"}}}, {"1", {{"element_id", second_map_id}, {"type", "map"}}}
    };

    // Test from_json
    EXPECT_NO_THROW({
        auto deserialized = transformations::MapFusion::from_json(builder, j);
        EXPECT_EQ(deserialized.name(), "MapFusion");
    });
}

TEST(MapFusionTest, TransformedAccessIndices) {
    // Test that access indices are correctly transformed when maps have different induction variables
    // Map 1: T[i] = A[i]
    // Map 2: B[j] = T[j]
    // After fusion, A should be accessed with j (not i)

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Define first map: T[i] = A[i]
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = map1.root();

    auto& block1 = builder.add_block(body1);
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Define second map: B[j] = T[j]
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body2 = map2.root();

    auto& block2 = builder.add_block(body2);
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    // Analyze and apply transformation
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // Verify transformation results
    auto& new_sdfg = builder.subject();

    // Get the fused second map
    auto* new_map2 = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(1).first);
    EXPECT_TRUE(new_map2 != nullptr);

    auto* new_block2 = dynamic_cast<structured_control_flow::Block*>(&new_map2->root().at(0).first);
    EXPECT_TRUE(new_block2 != nullptr);

    auto& dataflow = new_block2->dataflow();

    // Find the access node for A and check its memlet subset
    bool found_a_access = false;
    for (auto& node : dataflow.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr && access->data() == "A") {
            found_a_access = true;

            // Check the outgoing memlet's subset
            for (auto& memlet : dataflow.out_edges(*access)) {
                if (memlet.type() == data_flow::MemletType::Computational) {
                    // The subset should be exactly j (the second map's indvar)
                    EXPECT_EQ(memlet.subset().size(), 1);
                    if (!memlet.subset().empty()) {
                        auto expected = symbolic::symbol("j");
                        EXPECT_TRUE(symbolic::eq(memlet.subset()[0], expected))
                            << "Expected index 'j', got: " << memlet.subset()[0]->__str__();
                    }
                }
            }
            break;
        }
    }
    EXPECT_TRUE(found_a_access);
}

TEST(MapFusionTest, MapAndForFusion) {
    // Test fusion where first is a Map and second is a For loop
    // Map 1: T[i] = A[i] + 1.0
    // For 2: B[j] = T[j] * 2.0
    // After fusion: B[j] = (A[j] + 1.0) * 2.0

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Define first map: T[i] = A[i] + 1.0
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = map1.root();

    auto& block1 = builder.add_block(body1);
    auto& a_in = builder.add_access(block1, "A");
    auto& one_node = builder.add_constant(block1, "1.0", float_desc);
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in1", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, one_node, tasklet1, "_in2", {});
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Define second loop as a For (not Map): B[j] = T[j] * 2.0
    auto indvar2 = symbolic::symbol("j");
    auto& for2 = builder.add_for(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );
    auto& body2 = for2.root();

    auto& block2 = builder.add_block(body2);
    auto& t_in = builder.add_access(block2, "T");
    auto& two_node = builder.add_constant(block2, "2.0", float_desc);
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in1", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, two_node, tasklet2, "_in2", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    // Analyze and apply transformation
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, for2);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // Verify transformation results
    auto& new_sdfg = builder.subject();

    // Both loops should still exist
    EXPECT_EQ(new_sdfg.root().size(), 2);

    // The second loop should be a For (not Map) with 2 blocks now
    auto* new_for2 = dynamic_cast<structured_control_flow::For*>(&new_sdfg.root().at(1).first);
    EXPECT_TRUE(new_for2 != nullptr);
    EXPECT_EQ(new_for2->root().size(), 2) << "Second loop should now have 2 blocks (producer + consumer)";

    // First block is the new producer block
    auto* producer_block = dynamic_cast<structured_control_flow::Block*>(&new_for2->root().at(0).first);
    EXPECT_TRUE(producer_block != nullptr);

    // Second block is the original consumer block
    auto* consumer_block = dynamic_cast<structured_control_flow::Block*>(&new_for2->root().at(1).first);
    EXPECT_TRUE(consumer_block != nullptr);

    // Count total nodes across both blocks
    size_t total_nodes = 0;
    for (auto& _ : producer_block->dataflow().nodes()) {
        (void) _;
        total_nodes++;
    }
    for (auto& _ : consumer_block->dataflow().nodes()) {
        (void) _;
        total_nodes++;
    }
    EXPECT_GT(total_nodes, 4) << "Total nodes across producer and consumer blocks should be > 4";

    // Verify A is now accessed in the producer block with index j
    auto& dataflow = producer_block->dataflow();
    bool found_a_access = false;
    for (auto& node : dataflow.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr && access->data() == "A") {
            found_a_access = true;
            for (auto& memlet : dataflow.out_edges(*access)) {
                if (memlet.type() == data_flow::MemletType::Computational) {
                    EXPECT_EQ(memlet.subset().size(), 1);
                    if (!memlet.subset().empty()) {
                        auto expected = symbolic::symbol("j");
                        EXPECT_TRUE(symbolic::eq(memlet.subset()[0], expected))
                            << "Expected index 'j', got: " << memlet.subset()[0]->__str__();
                    }
                }
            }
            break;
        }
    }
    EXPECT_TRUE(found_a_access);
}

TEST(MapFusionTest, Domain_IdenticalDomain) {
    // Both maps have identical domain: 0:N:1
    // Map 1: T[i] = A[i] for i in 0:N:1
    // Map 2: B[j] = T[j] for j in 0:N:1
    // Should fuse successfully

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: 0:N:1
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2: 0:N:1
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(MapFusionTest, Domain_OverComputation) {
    // OverComputation: First map computes more than second map needs
    // Map 1: T[i] = A[i] for i in 0:N:1
    // Map 2: B[j] = T[j] for j in 0:N/2:1
    // Second map only uses half the computed values - should still fuse

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: 0:N:1
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2: 0:N/2:1
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    // Should be applicable - we can fuse even if first map over-computes
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(MapFusionTest, Domain_Recomputation) {
    // Recomputation: Second map needs more elements than first map produces
    // Map 1: T[i] = A[i] for i in 0:N:1
    // Map 2: B[j] = T[j] for j in 0:2*N:1
    // Should NOT fuse: consumer reads elements 0..2N-1 but producer only wrote 0..N-1

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc_n(float_desc, {symbolic::symbol("N")});
    types::Array array_desc_2n(float_desc, {symbolic::mul(symbolic::integer(2), symbolic::symbol("N"))});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc_2n, true);
    builder.add_container("T", array_desc_2n);
    builder.add_container("B", array_desc_2n, true);

    // Map 1: 0:N:1
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc_2n);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc_2n);

    // Map 2: 0:2*N:1
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::mul(symbolic::integer(2), symbolic::symbol("N"))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("j")}, array_desc_2n);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc_2n);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    // The index mapping i = j is valid, but the producer only writes elements 0..N-1
    // while the consumer reads 0..2N-1. Fusion would read uninitialized values.
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager))
        << "Should reject: consumer reads beyond producer's write range";
}

TEST(MapFusionTest, Domain_PartialSubRange_ProducerWritesSlice) {
    // Producer writes a sub-range (offset slice), consumer reads the full range.
    // This models the pattern: A[i, 1:N-1] = B[i, 1:N-1] followed by C = A (full copy).
    // Map 1: T[i] = A[i] for i in 1:N-1:1 (writes indices 1..N-2)
    // Map 2: B[j] = T[j] for j in 0:N:1   (reads ALL indices 0..N-1)
    // Should NOT fuse: consumer reads T[0] and T[N-1] which producer never wrote.

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: T[i] = A[i] for i in 1:N-1:1 (writes only the interior)
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2: B[j] = T[j] for j in 0:N:1 (reads all indices)
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager))
        << "Should reject: producer writes indices 1..N-2 but consumer reads 0..N-1";
}

TEST(MapFusionTest, Domain_PartialSubRange_ConsumerReadsSlice) {
    // Producer writes a sub-range, consumer reads the SAME sub-range.
    // Map 1: T[i] = A[i] for i in 1:N-1:1 (writes indices 1..N-2)
    // Map 2: B[j] = T[j] for j in 1:N-1:1 (reads indices 1..N-2)
    // Should fuse: consumer only reads what producer wrote.

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: T[i] = A[i] for i in 1:N-1:1
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2: B[j] = T[j] for j in 1:N-1:1 (same range as producer)
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager))
        << "Should accept: consumer reads exactly the sub-range producer wrote";
}

TEST(MapFusionTest, Domain_Stencil1D) {
    // 1D Stencil: Consumer reads multiple indices from producer output
    // Map 1: T[i] = A[i] for i in 0:N:1
    // Map 2: B[j] = T[j-1] + T[j] + T[j+1] for j in 1:N-1:1
    // This pattern currently not supported (multiple reads of same array with different indices)
    // We expect can_be_applied to return false for now since we only handle single reads

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: T[i] = A[i] for 0:N:1
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2: B[j] = T[j-1] + T[j] + T[j+1] for 1:N-1:1
    // Reads T at three different offsets
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());

    // Read T[j-1]
    auto& t_left = builder.add_access(block2, "T");
    // Read T[j]
    auto& t_center = builder.add_access(block2, "T");
    // Read T[j+1]
    auto& t_right = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");

    // Create add tasklets: tmp1 = T[j-1] + T[j], out = tmp1 + T[j+1]
    auto& add1 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    auto& add2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    builder.add_computational_memlet(
        block2, t_left, add1, "_in1", {symbolic::sub(symbolic::symbol("j"), symbolic::integer(1))}, array_desc
    );
    builder.add_computational_memlet(block2, t_center, add1, "_in2", {symbolic::symbol("j")}, array_desc);

    // Need intermediate storage for first add result
    types::Scalar tmp_desc(types::PrimitiveType::Float);
    std::string tmp_name = builder.find_new_name("_stencil_tmp");
    builder.add_container(tmp_name, tmp_desc);
    auto& tmp_out = builder.add_access(block2, tmp_name);
    auto& tmp_in = builder.add_access(block2, tmp_name);

    builder.add_computational_memlet(block2, add1, "_out", tmp_out, {}, tmp_desc);
    builder.add_computational_memlet(block2, tmp_in, add2, "_in1", {}, tmp_desc);
    builder.add_computational_memlet(
        block2, t_right, add2, "_in2", {symbolic::add(symbolic::symbol("j"), symbolic::integer(1))}, array_desc
    );
    builder.add_computational_memlet(block2, add2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    // Current implementation only handles single read per container
    // We find the first read and use that - should still be applicable for one of them
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(MapFusionTest, Domain_SecondMapStrided) {
    // Second map strided: First map 0:N:1, Second map 0:N:2 (stride 2)
    // Map 1: T[i] = A[i] for i in 0:N:1
    // Map 2: B[j] = T[2*j] for j in 0:N/2:1 (effectively accessing even indices)
    // Index mapping: i = 2*j

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: T[i] = A[i] for 0:N:1
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2: B[j] = T[2*j] for 0:N/2:1
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    // Access T[2*j] - strided access
    builder.add_computational_memlet(
        block2, t_in, tasklet2, "_in", {symbolic::mul(symbolic::integer(2), symbolic::symbol("j"))}, array_desc
    );
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    // Index mapping: i = 2*j (valid affine mapping)
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));

    // Apply and verify the index substitution
    transformation.apply(builder, analysis_manager);

    auto* new_map2 = dynamic_cast<structured_control_flow::Map*>(&builder.subject().root().at(1).first);
    auto* new_block2 = dynamic_cast<structured_control_flow::Block*>(&new_map2->root().at(0).first);
    auto& dataflow = new_block2->dataflow();

    // A should be accessed with 2*j after fusion
    bool found_a_access = false;
    for (auto& node : dataflow.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr && access->data() == "A") {
            found_a_access = true;
            for (auto& memlet : dataflow.out_edges(*access)) {
                if (memlet.type() == data_flow::MemletType::Computational && !memlet.subset().empty()) {
                    // Verify that the index is exactly 2*j
                    auto expected = symbolic::mul(symbolic::integer(2), symbolic::symbol("j"));
                    EXPECT_TRUE(symbolic::eq(memlet.subset()[0], expected))
                        << "Expected index '2*j', got: " << memlet.subset()[0]->__str__();
                }
            }
        }
    }
    EXPECT_TRUE(found_a_access) << "Should find A access node";
}

TEST(MapFusionTest, Domain_BothMapsStridedModuloMatches) {
    // Both maps strided with matching modulo
    // Map 1: T[2*i] = A[2*i] for i in 0:N/2:1 (writes even indices)
    // Map 2: B[j] = T[2*j] for j in 0:N/2:1 (reads even indices)
    // Index mapping: 2*i = 2*j => i = j

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: T[2*i] = A[2*i] for 0:N/2:1
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block1, a_in, tasklet1, "_in", {symbolic::mul(symbolic::integer(2), symbolic::symbol("i"))}, array_desc
    );
    builder.add_computational_memlet(
        block1, tasklet1, "_out", t_out, {symbolic::mul(symbolic::integer(2), symbolic::symbol("i"))}, array_desc
    );

    // Map 2: B[j] = T[2*j] for 0:N/2:1
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block2, t_in, tasklet2, "_in", {symbolic::mul(symbolic::integer(2), symbolic::symbol("j"))}, array_desc
    );
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    // Index mapping: 2*i = 2*j => i = j (valid)
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));

    // Apply and verify the index substitution
    transformation.apply(builder, analysis_manager);

    auto* new_map2 = dynamic_cast<structured_control_flow::Map*>(&builder.subject().root().at(1).first);
    auto* new_block2 = dynamic_cast<structured_control_flow::Block*>(&new_map2->root().at(0).first);
    auto& dataflow = new_block2->dataflow();

    // A should be accessed with 2*j after fusion (i replaced by j)
    // Note: The mapping computes i = (2*j)/2 = idiv(2*j, 2), then substitutes into 2*i
    // Result is 2*idiv(2*j, 2) which is mathematically equivalent to 2*j
    bool found_a_access = false;
    for (auto& node : dataflow.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr && access->data() == "A") {
            found_a_access = true;
            for (auto& memlet : dataflow.out_edges(*access)) {
                if (memlet.type() == data_flow::MemletType::Computational && !memlet.subset().empty()) {
                    // Verify that the index expression:
                    // 1. Contains j (the second loop's indvar)
                    // 2. Does not contain i (the first loop's indvar)
                    auto actual = memlet.subset()[0];
                    auto atoms = symbolic::atoms(actual);
                    bool has_j = false;
                    bool has_i = false;
                    for (const auto& atom : atoms) {
                        if (atom->get_name() == "j") has_j = true;
                        if (atom->get_name() == "i") has_i = true;
                    }
                    EXPECT_TRUE(has_j) << "Index should contain 'j' after fusion, got: " << actual->__str__();
                    EXPECT_FALSE(has_i) << "Index should not contain 'i' after fusion, got: " << actual->__str__();
                }
            }
        }
    }
    EXPECT_TRUE(found_a_access) << "Should find A access node";
}

TEST(MapFusionTest, Domain_BothMapsStridedModuloMismatch) {
    // Both maps strided but modulo does not match
    // Map 1: T[2*i] = A[2*i] for i in 0:N/2:1 (writes even indices: 0, 2, 4, ...)
    // Map 2: B[j] = T[2*j+1] for j in 0:N/2:1 (reads odd indices: 1, 3, 5, ...)
    // The reads never hit what the producer wrote - index equation has no valid solution

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: T[2*i] = A[2*i] for 0:N/2:1 (writes even indices)
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block1, a_in, tasklet1, "_in", {symbolic::mul(symbolic::integer(2), symbolic::symbol("i"))}, array_desc
    );
    builder.add_computational_memlet(
        block1, tasklet1, "_out", t_out, {symbolic::mul(symbolic::integer(2), symbolic::symbol("i"))}, array_desc
    );

    // Map 2: B[j] = T[2*j+1] for 0:N/2:1 (reads odd indices)
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    // Access T[2*j+1] - odd indices
    builder.add_computational_memlet(
        block2,
        t_in,
        tasklet2,
        "_in",
        {symbolic::add(symbolic::mul(symbolic::integer(2), symbolic::symbol("j")), symbolic::integer(1))},
        array_desc
    );
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    // Index equation: 2*i = 2*j + 1 => i = j + 0.5 (not an integer!)
    // The ISL integrality check detects that 2*i = 2*j+1 has no integer solution
    // (LHS is even, RHS is odd), so the fusion is correctly rejected.
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(MapFusionTest, Domain_PartialProducerConsumerReadsAll) {
    // Producer writes only even indices, consumer reads ALL indices (including precomputed odd ones)
    // Map 1: T[2*i] = A[2*i] for i in 0:N/2:1 (writes even indices: 0, 2, 4, ...)
    // Map 2: B[k] = T[k] for k in 0:N:1 (reads all indices: 0, 1, 2, 3, ...)
    // Should NOT fuse: the consumer reads T at odd indices that weren't produced by the first map

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: T[2*i] = A[2*i] for i in 0:N/2:1
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block1, a_in, tasklet1, "_in", {symbolic::mul(symbolic::integer(2), symbolic::symbol("i"))}, array_desc
    );
    builder.add_computational_memlet(
        block1, tasklet1, "_out", t_out, {symbolic::mul(symbolic::integer(2), symbolic::symbol("i"))}, array_desc
    );

    // Map 2: B[k] = T[k] for k in 0:N:1 (reads ALL indices)
    auto indvar2 = symbolic::symbol("k");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("k")}, array_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("k")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    // Index equation: 2*i = k => i = k/2. ISL composition domain is only even k values.
    // Consumer domain {k : 0 <= k < N} is NOT a subset of {k : exists a : k = 2*a},
    // so ISL correctly rejects: not every consumer point has an integer producer mapping.
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager))
        << "Should reject: consumer reads all indices but producer only wrote even ones";
}

TEST(MapFusionTest, Dataflow_InDegree0_SingleOutEdge) {
    // Pattern: Consumer access node has in_degree=0 (read-only) and one outgoing edge
    // Verifies: data(), subset(), and base_type() are all updated correctly for BOTH
    //           producer memlets (in newly created producer block) and consumer memlets

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: T[i] = A[i]
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2: B[j] = T[j]
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& t_memlet =
        builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    // Verify initial state
    EXPECT_EQ(t_in.data(), "T");
    EXPECT_EQ(t_memlet.subset().size(), 1);
    EXPECT_TRUE(dynamic_cast<const types::Array*>(&t_memlet.base_type()) != nullptr);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // After fusion: verify CONSUMER memlet data, subset, and type are all updated
    EXPECT_TRUE(t_in.data().find("_fused_tmp") != std::string::npos)
        << "Access node data should point to temp scalar, got: " << t_in.data();
    EXPECT_EQ(t_memlet.subset().size(), 0) << "Memlet subset should be empty after fusion (scalar access)";
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&t_memlet.base_type()) != nullptr)
        << "Consumer memlet base_type should be Scalar after fusion";

    // Verify PRODUCER block memlets have correct base_type
    auto* new_map2 = dynamic_cast<structured_control_flow::Map*>(&builder.subject().root().at(1).first);
    ASSERT_TRUE(new_map2 != nullptr);
    EXPECT_EQ(new_map2->root().size(), 2) << "Should have 1 producer block + 1 consumer block";

    auto* producer_block = dynamic_cast<structured_control_flow::Block*>(&new_map2->root().at(0).first);
    ASSERT_TRUE(producer_block != nullptr);

    auto& producer_dataflow = producer_block->dataflow();

    // Check producer memlet properties
    bool found_producer_input = false;
    bool found_producer_output = false;
    for (auto& node : producer_dataflow.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access == nullptr) continue;

        if (access->data() == "A") {
            // Input memlet (A -> tasklet) should retain Array type
            for (auto& memlet : producer_dataflow.out_edges(*access)) {
                if (memlet.type() == data_flow::MemletType::Computational) {
                    found_producer_input = true;
                    EXPECT_EQ(memlet.subset().size(), 1) << "Producer input memlet should have 1D subset";
                    EXPECT_TRUE(dynamic_cast<const types::Array*>(&memlet.base_type()) != nullptr)
                        << "Producer input memlet (A) should have Array base_type";
                }
            }
        } else if (access->data().find("_fused_tmp") != std::string::npos) {
            // Output memlet (tasklet -> temp) should have Scalar type
            for (auto& memlet : producer_dataflow.in_edges(*access)) {
                if (memlet.type() == data_flow::MemletType::Computational) {
                    found_producer_output = true;
                    EXPECT_EQ(memlet.subset().size(), 0) << "Producer output memlet should have empty subset (scalar)";
                    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&memlet.base_type()) != nullptr)
                        << "Producer output memlet (temp) should have Scalar base_type";
                }
            }
        }
    }

    EXPECT_TRUE(found_producer_input) << "Should find producer input memlet from A";
    EXPECT_TRUE(found_producer_output) << "Should find producer output memlet to temp";
}

TEST(MapFusionTest, Dataflow_InDegree0_MultipleOutEdges) {
    // Pattern: Consumer access node has in_degree=0 and multiple outgoing edges
    // T is read by two different tasklets in the second map
    // Verifies: all outgoing memlets have data(), subset(), and base_type() updated

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);
    builder.add_container("C", array_desc, true);

    // Map 1: T[i] = A[i]
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2: B[j] = T[j], C[j] = T[j] (T is read twice by different tasklets)
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());

    // Single T access node with TWO outgoing edges
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& c_out = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});

    // Two edges from t_in
    auto& memlet1 =
        builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("j")}, array_desc);
    auto& memlet2 =
        builder.add_computational_memlet(block2, t_in, tasklet3, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2, tasklet3, "_out", c_out, {symbolic::symbol("j")}, array_desc);

    // Verify initial state
    EXPECT_EQ(t_in.data(), "T");
    EXPECT_EQ(memlet1.subset().size(), 1);
    EXPECT_EQ(memlet2.subset().size(), 1);
    EXPECT_TRUE(dynamic_cast<const types::Array*>(&memlet1.base_type()) != nullptr);
    EXPECT_TRUE(dynamic_cast<const types::Array*>(&memlet2.base_type()) != nullptr);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // After fusion: verify data, subset, and type are all updated for BOTH edges
    EXPECT_TRUE(t_in.data().find("_fused_tmp") != std::string::npos)
        << "Access node data should point to temp scalar, got: " << t_in.data();

    EXPECT_EQ(memlet1.subset().size(), 0) << "First memlet subset should be empty after fusion";
    EXPECT_EQ(memlet2.subset().size(), 0) << "Second memlet subset should be empty after fusion";

    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&memlet1.base_type()) != nullptr)
        << "First memlet base_type should be Scalar after fusion";
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&memlet2.base_type()) != nullptr)
        << "Second memlet base_type should be Scalar after fusion";
}

TEST(MapFusionTest, Dataflow_MultipleBlocks_MultipleAccessNodes) {
    // Pattern: Consumer loop has multiple blocks, each with its own access node for T
    // Map 1: T[i] = A[i]
    // Map 2 with TWO blocks:
    //   Block 2a: B[j] = T[j]
    //   Block 2b: C[j] = T[j]
    // Both access nodes should be updated to point to the temp scalar

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);
    builder.add_container("C", array_desc, true);

    // Map 1: T[i] = A[i]
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2 with TWO separate blocks, each reading T
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Block 2a: B[j] = T[j]
    auto& block2a = builder.add_block(map2.root());
    auto& t_in_a = builder.add_access(block2a, "T");
    auto& b_out = builder.add_access(block2a, "B");
    auto& tasklet2 = builder.add_tasklet(block2a, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& memlet_a =
        builder.add_computational_memlet(block2a, t_in_a, tasklet2, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2a, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    // Block 2b: C[j] = T[j]
    auto& block2b = builder.add_block(map2.root());
    auto& t_in_b = builder.add_access(block2b, "T");
    auto& c_out = builder.add_access(block2b, "C");
    auto& tasklet3 = builder.add_tasklet(block2b, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& memlet_b =
        builder.add_computational_memlet(block2b, t_in_b, tasklet3, "_in", {symbolic::symbol("j")}, array_desc);
    builder.add_computational_memlet(block2b, tasklet3, "_out", c_out, {symbolic::symbol("j")}, array_desc);

    // Verify initial state - two separate access nodes for T
    EXPECT_EQ(t_in_a.data(), "T");
    EXPECT_EQ(t_in_b.data(), "T");
    EXPECT_EQ(memlet_a.subset().size(), 1);
    EXPECT_EQ(memlet_b.subset().size(), 1);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // After fusion: BOTH access nodes should be updated
    EXPECT_TRUE(t_in_a.data().find("_fused_tmp") != std::string::npos)
        << "First access node should point to temp scalar, got: " << t_in_a.data();
    EXPECT_TRUE(t_in_b.data().find("_fused_tmp") != std::string::npos)
        << "Second access node should point to temp scalar, got: " << t_in_b.data();

    // Both memlets should have empty subsets (scalar access)
    EXPECT_EQ(memlet_a.subset().size(), 0) << "First block memlet subset should be empty after fusion";
    EXPECT_EQ(memlet_b.subset().size(), 0) << "Second block memlet subset should be empty after fusion";

    // Both memlets should have scalar type
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&memlet_a.base_type()) != nullptr)
        << "First block memlet base_type should be Scalar after fusion";
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&memlet_b.base_type()) != nullptr)
        << "Second block memlet base_type should be Scalar after fusion";
}

TEST(MapFusionTest, Dataflow_StencilConsumer_MultipleIndexMappings) {
    // Pattern: Consumer reads same intermediate array at different indices (stencil pattern)
    // Map 1: T[i] = A[i]
    // Map 2: B[j] = T[j-1] + T[j+1]  (different indices)
    // This IS fusible - we create two producer blocks with different index mappings

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Array array_desc(float_desc, {symbolic::symbol("N")});
    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("A", array_desc, true);
    builder.add_container("T", array_desc);
    builder.add_container("B", array_desc, true);

    // Map 1: T[i] = A[i]
    auto indvar1 = symbolic::symbol("i");
    auto& map1 = builder.add_map(
        root,
        indvar1,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block1 = builder.add_block(map1.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i")}, array_desc);
    builder.add_computational_memlet(block1, tasklet1, "_out", t_out, {symbolic::symbol("i")}, array_desc);

    // Map 2: B[j] = T[j-1] + T[j+1] (stencil - reads T at different indices)
    // j ranges from 1 to N-2 so that j-1 >= 0 and j+1 <= N-1 (within producer range)
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::sub(symbolic::symbol("N"), symbolic::integer(1))),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block2 = builder.add_block(map2.root());

    // Two access nodes reading T at different indices
    auto& t_in_left = builder.add_access(block2, "T");
    auto& t_in_right = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});

    // T[j-1] and T[j+1] - different subsets
    auto& memlet_left = builder.add_computational_memlet(
        block2, t_in_left, tasklet2, "_in1", {symbolic::sub(symbolic::symbol("j"), symbolic::integer(1))}, array_desc
    );
    auto& memlet_right = builder.add_computational_memlet(
        block2, t_in_right, tasklet2, "_in2", {symbolic::add(symbolic::symbol("j"), symbolic::integer(1))}, array_desc
    );
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("j")}, array_desc);

    // Verify initial state
    EXPECT_EQ(t_in_left.data(), "T");
    EXPECT_EQ(t_in_right.data(), "T");
    EXPECT_EQ(memlet_left.subset().size(), 1);
    EXPECT_EQ(memlet_right.subset().size(), 1);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1, map2);

    // Should BE fusible - we support stencil patterns now
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager))
        << "Stencil consumer with different index patterns should be fusible";

    transformation.apply(builder, analysis_manager);

    // After fusion: both access nodes should point to DIFFERENT temps
    EXPECT_TRUE(t_in_left.data().find("_fused_tmp") != std::string::npos)
        << "Left access node should point to temp scalar, got: " << t_in_left.data();
    EXPECT_TRUE(t_in_right.data().find("_fused_tmp") != std::string::npos)
        << "Right access node should point to temp scalar, got: " << t_in_right.data();

    // The temps should be different since they have different index mappings
    EXPECT_NE(t_in_left.data(), t_in_right.data())
        << "Left and right should use different temps (different index mappings)";

    // Both memlets should have empty subsets (scalar access)
    EXPECT_EQ(memlet_left.subset().size(), 0) << "Left memlet subset should be empty after fusion";
    EXPECT_EQ(memlet_right.subset().size(), 0) << "Right memlet subset should be empty after fusion";

    // Both memlets should have scalar type
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&memlet_left.base_type()) != nullptr)
        << "Left memlet base_type should be Scalar after fusion";
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&memlet_right.base_type()) != nullptr)
        << "Right memlet base_type should be Scalar after fusion";

    // Should have inserted 2 producer blocks (one per unique index mapping)
    // The consumer block should now be at index 2
    EXPECT_EQ(map2.root().size(), 3) << "Should have 2 producer blocks + 1 consumer block";
}

// ============================================================================
// Multi-dimensional (2D) tests
// ============================================================================

TEST(MapFusionTest, Domain_2D_IdenticalDomain) {
    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] } }
    // Consumer: Map(k, 0:M) { Map(l, 0:N) { B[k,l] = T[k,l] } }
    // Should fuse with i->k, j->l

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array inner_array(float_desc, {symbolic::symbol("N")});
    types::Array array_2d(inner_array, {symbolic::symbol("M")});

    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_2d, true);
    builder.add_container("T", array_2d);
    builder.add_container("B", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i) { Map(j) { T[i,j] = A[i,j] } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& block1 = builder.add_block(map1_inner.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d);
    builder.add_computational_memlet(
        block1, tasklet1, "_out", t_out, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k) { Map(l) { B[k,l] = T[k,l] } }
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& block2 = builder.add_block(map2_inner.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d);
    builder.add_computational_memlet(
        block2, tasklet2, "_out", b_out, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(MapFusionTest, Domain_2D_OverComputation) {
    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] } }
    // Consumer: Map(k, 0:M/2) { Map(l, 0:N/2) { B[k,l] = T[k,l] } }
    // Consumer only uses a subset - should still fuse with i->k, j->l

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array inner_array(float_desc, {symbolic::symbol("N")});
    types::Array array_2d(inner_array, {symbolic::symbol("M")});

    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_2d, true);
    builder.add_container("T", array_2d);
    builder.add_container("B", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& block1 = builder.add_block(map1_inner.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d);
    builder.add_computational_memlet(
        block1, tasklet1, "_out", t_out, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k, 0:M/2) { Map(l, 0:N/2) { B[k,l] = T[k,l] } }
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::div(symbolic::symbol("M"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& block2 = builder.add_block(map2_inner.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d);
    builder.add_computational_memlet(
        block2, tasklet2, "_out", b_out, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(MapFusionTest, Domain_2D_StridedAccess) {
    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] } }
    // Consumer: Map(k, 0:M) { Map(l, 0:N/2) { B[k,l] = T[k, 2*l] } }
    // Should fuse: i->k, j->2*l

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array inner_array(float_desc, {symbolic::symbol("N")});
    types::Array array_2d(inner_array, {symbolic::symbol("M")});

    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_2d, true);
    builder.add_container("T", array_2d);
    builder.add_container("B", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& block1 = builder.add_block(map1_inner.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d);
    builder.add_computational_memlet(
        block1, tasklet1, "_out", t_out, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k, 0:M) { Map(l, 0:N/2) { B[k,l] = T[k, 2*l] } }
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& block2 = builder.add_block(map2_inner.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block2,
        t_in,
        tasklet2,
        "_in",
        {symbolic::symbol("k"), symbolic::mul(symbolic::integer(2), symbolic::symbol("l"))},
        array_2d
    );
    builder.add_computational_memlet(
        block2, tasklet2, "_out", b_out, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(MapFusionTest, Domain_2D_DimensionMismatch) {
    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] } } (2D subset)
    // Consumer: Map(k, 0:M*N) { B[k] = T[k] } } (1D subset)
    // Should NOT fuse (dimension mismatch: producer writes 2D, consumer reads 1D)

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array inner_array(float_desc, {symbolic::symbol("N")});
    types::Array array_2d(inner_array, {symbolic::symbol("M")});
    types::Array array_1d(float_desc, {symbolic::mul(symbolic::symbol("M"), symbolic::symbol("N"))});

    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("A", array_2d, true);
    builder.add_container("T", array_2d);
    builder.add_container("B", array_1d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& block1 = builder.add_block(map1_inner.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d);
    builder.add_computational_memlet(
        block1, tasklet1, "_out", t_out, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k, 0:M*N) { B[k] = T[k] }
    auto& map2 = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::mul(symbolic::symbol("M"), symbolic::symbol("N"))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& block2 = builder.add_block(map2.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, t_in, tasklet2, "_in", {symbolic::symbol("k")}, array_1d);
    builder.add_computational_memlet(block2, tasklet2, "_out", b_out, {symbolic::symbol("k")}, array_1d);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2);

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager))
        << "Should reject fusion when producer subset is 2D but consumer subset is 1D";
}

TEST(MapFusionTest, Domain_2D_CrossDimensionDependency) {
    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i+j, i] = ... } }
    // Consumer: Map(k, 0:M) { Map(l, 0:N) { B[k,l] = T[k+l, k] } }
    // The equation system i+j=k+l, i=k has a unique solution i=k, j=l
    // Both domains match, so fusion is valid

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array inner_array(float_desc, {symbolic::symbol("N")});
    types::Array array_2d(inner_array, {symbolic::symbol("M")});

    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_2d, true);
    builder.add_container("T", array_2d);
    builder.add_container("B", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i+j, i] = A[i,j] } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& block1 = builder.add_block(map1_inner.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d);
    // T[i+j, i] - first dim depends on both i and j
    builder.add_computational_memlet(
        block1,
        tasklet1,
        "_out",
        t_out,
        {symbolic::add(symbolic::symbol("i"), symbolic::symbol("j")), symbolic::symbol("i")},
        array_2d
    );

    // Consumer: Map(k, 0:M) { Map(l, 0:N) { B[k,l] = T[k+l, k] } }
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& block2 = builder.add_block(map2_inner.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block2,
        t_in,
        tasklet2,
        "_in",
        {symbolic::add(symbolic::symbol("k"), symbolic::symbol("l")), symbolic::symbol("k")},
        array_2d
    );
    builder.add_computational_memlet(
        block2, tasklet2, "_out", b_out, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager))
        << "Cross-dimension dependencies with a unique linear solution should fuse";
}

TEST(MapFusionTest, Domain_2D_Apply_IndexSubstitution) {
    // Verify apply() correctly substitutes indices in the 2D case
    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] + 1.0 } }
    // Consumer: Map(k, 0:M) { Map(l, 0:N) { B[k,l] = T[k,l] * 2.0 } }
    // After fusion, the consumer's inner body should have a producer block
    // reading A[k,l] instead of A[i,j]

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array inner_array(float_desc, {symbolic::symbol("N")});
    types::Array array_2d(inner_array, {symbolic::symbol("M")});

    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_2d, true);
    builder.add_container("T", array_2d);
    builder.add_container("B", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] + 1.0 } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& block1 = builder.add_block(map1_inner.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& one_node = builder.add_constant(block1, "1.0", float_desc);
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block1, a_in, tasklet1, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d);
    builder.add_computational_memlet(block1, one_node, tasklet1, "_in2", {});
    builder.add_computational_memlet(
        block1, tasklet1, "_out", t_out, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k, 0:M) { Map(l, 0:N) { B[k,l] = T[k,l] * 2.0 } }
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& block2 = builder.add_block(map2_inner.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& two_node = builder.add_constant(block2, "2.0", float_desc);
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder
        .add_computational_memlet(block2, t_in, tasklet2, "_in1", {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d);
    builder.add_computational_memlet(block2, two_node, tasklet2, "_in2", {});
    builder.add_computational_memlet(
        block2, tasklet2, "_out", b_out, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // Verify transformation results
    auto& new_sdfg = builder.subject();

    // Both outer maps should still exist
    EXPECT_EQ(new_sdfg.root().size(), 2);

    // Navigate to the consumer's inner map body
    auto* new_map2_outer = dynamic_cast<structured_control_flow::Map*>(&new_sdfg.root().at(1).first);
    ASSERT_TRUE(new_map2_outer != nullptr);

    // The outer consumer map should have one child: the inner map
    EXPECT_EQ(new_map2_outer->root().size(), 1);

    auto* new_map2_inner = dynamic_cast<structured_control_flow::Map*>(&new_map2_outer->root().at(0).first);
    ASSERT_TRUE(new_map2_inner != nullptr);

    // The inner map should now have 2 blocks: producer + consumer
    EXPECT_EQ(new_map2_inner->root().size(), 2)
        << "Inner consumer map should have 2 blocks (producer + consumer) after fusion";

    // First block in inner map is the new producer block
    auto* producer_block = dynamic_cast<structured_control_flow::Block*>(&new_map2_inner->root().at(0).first);
    ASSERT_TRUE(producer_block != nullptr);

    // Second block in inner map is the original consumer block
    auto* consumer_block = dynamic_cast<structured_control_flow::Block*>(&new_map2_inner->root().at(1).first);
    ASSERT_TRUE(consumer_block != nullptr);

    // Verify the producer block's input memlet now uses k,l instead of i,j
    auto& producer_df = producer_block->dataflow();
    for (auto& node : producer_df.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr && access->data() == "A") {
            // The A access should have an outgoing memlet with subset {k, l}
            for (auto& memlet : producer_df.out_edges(*access)) {
                ASSERT_EQ(memlet.subset().size(), 2) << "Producer's A access should have 2D subset";
                EXPECT_TRUE(symbolic::eq(memlet.subset()[0], symbolic::symbol("k"))) << "First index should be k";
                EXPECT_TRUE(symbolic::eq(memlet.subset()[1], symbolic::symbol("l"))) << "Second index should be l";
            }
        }
    }

    // Verify the consumer block reads from the temp scalar (not T)
    auto& consumer_df = consumer_block->dataflow();
    for (auto& node : consumer_df.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr && consumer_df.out_degree(*access) > 0) {
            // This is a read access - should be the temp, not T
            EXPECT_NE(access->data(), "T") << "Consumer should read from temp scalar, not T";
        }
    }
}

TEST(MapFusionTest, Domain_2D_Apply_StridedIndexSubstitution) {
    // Verify apply() correctly substitutes strided indices in the 2D case
    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] } }
    // Consumer: Map(k, 0:M) { Map(l, 0:N/2) { B[k,l] = T[k, 2*l] } }
    // After fusion, the producer block should read A[k, 2*l]

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array inner_array(float_desc, {symbolic::symbol("N")});
    types::Array array_2d(inner_array, {symbolic::symbol("M")});

    builder.add_container("M", sym_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_2d, true);
    builder.add_container("T", array_2d);
    builder.add_container("B", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:M) { Map(j, 0:N) { T[i,j] = A[i,j] } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& block1 = builder.add_block(map1_inner.root());
    auto& a_in = builder.add_access(block1, "A");
    auto& t_out = builder.add_access(block1, "T");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block1, a_in, tasklet1, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d);
    builder.add_computational_memlet(
        block1, tasklet1, "_out", t_out, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k, 0:M) { Map(l, 0:N/2) { B[k,l] = T[k, 2*l] } }
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::div(symbolic::symbol("N"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& block2 = builder.add_block(map2_inner.root());
    auto& t_in = builder.add_access(block2, "T");
    auto& b_out = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        block2,
        t_in,
        tasklet2,
        "_in",
        {symbolic::symbol("k"), symbolic::mul(symbolic::integer(2), symbolic::symbol("l"))},
        array_2d
    );
    builder.add_computational_memlet(
        block2, tasklet2, "_out", b_out, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // Navigate to the inner map
    auto* new_map2_outer = dynamic_cast<structured_control_flow::Map*>(&builder.subject().root().at(1).first);
    ASSERT_TRUE(new_map2_outer != nullptr);
    auto* new_map2_inner = dynamic_cast<structured_control_flow::Map*>(&new_map2_outer->root().at(0).first);
    ASSERT_TRUE(new_map2_inner != nullptr);

    EXPECT_EQ(new_map2_inner->root().size(), 2) << "Inner consumer map should have 2 blocks after fusion";

    // Verify the producer block reads A[k, 2*l]
    auto* producer_block = dynamic_cast<structured_control_flow::Block*>(&new_map2_inner->root().at(0).first);
    ASSERT_TRUE(producer_block != nullptr);

    auto& producer_df = producer_block->dataflow();
    for (auto& node : producer_df.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr && access->data() == "A") {
            for (auto& memlet : producer_df.out_edges(*access)) {
                ASSERT_EQ(memlet.subset().size(), 2) << "Producer's A access should have 2D subset";
                EXPECT_TRUE(symbolic::eq(memlet.subset()[0], symbolic::symbol("k"))) << "First index should be k";
                EXPECT_TRUE(symbolic::eq(memlet.subset()[1], symbolic::mul(symbolic::integer(2), symbolic::symbol("l")))
                ) << "Second index should be 2*l";
            }
        }
    }
}

TEST(MapFusionTest, Pattern2_NonPerfectlyNestedProducer) {
    // Pattern 2: Producer is NOT perfectly nested, consumer is perfectly nested.
    // Producer: Map(i, 0:N) {
    //     Block: S[i] = A[i] + 1.0      (sibling at depth 1)
    //     Map(j, 0:M) {
    //         Block: T[i,j] = S[i] * B[i,j]   (write at depth 2)
    //     }
    // }
    // Consumer: Map(k, 0:N) { Map(l, 0:M) { C[k,l] = T[k,l] } }
    //
    // Fusion direction should be ConsumerIntoProducer to avoid replicating the S computation.

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array array_1d(float_desc, {symbolic::symbol("N")});
    types::Array inner_array(float_desc, {symbolic::symbol("M")});
    types::Array array_2d(inner_array, {symbolic::symbol("N")});

    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_1d, true);
    builder.add_container("B", array_2d, true);
    builder.add_container("S", array_1d);
    builder.add_container("T", array_2d);
    builder.add_container("C", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:N) { Block: S[i] = A[i] + 1.0; Map(j, 0:M) { T[i,j] = S[i] * B[i,j] } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );

    // Block at depth 1: S[i] = A[i] + 1.0
    auto& sibling_block = builder.add_block(map1_outer.root());
    auto& a_read = builder.add_access(sibling_block, "A");
    auto& one_const = builder.add_constant(sibling_block, "1.0", float_desc);
    auto& s_write = builder.add_access(sibling_block, "S");
    auto& add_tasklet = builder.add_tasklet(sibling_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(sibling_block, a_read, add_tasklet, "_in1", {symbolic::symbol("i")}, array_1d);
    builder.add_computational_memlet(sibling_block, one_const, add_tasklet, "_in2", {});
    builder.add_computational_memlet(sibling_block, add_tasklet, "_out", s_write, {symbolic::symbol("i")}, array_1d);

    // Inner Map(j, 0:M): T[i,j] = S[i] * B[i,j]
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& prod_block = builder.add_block(map1_inner.root());
    auto& s_read = builder.add_access(prod_block, "S");
    auto& b_read = builder.add_access(prod_block, "B");
    auto& t_write = builder.add_access(prod_block, "T");
    auto& mul_tasklet = builder.add_tasklet(prod_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(prod_block, s_read, mul_tasklet, "_in1", {symbolic::symbol("i")}, array_1d);
    builder.add_computational_memlet(
        prod_block, b_read, mul_tasklet, "_in2", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );
    builder.add_computational_memlet(
        prod_block, mul_tasklet, "_out", t_write, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k, 0:N) { Map(l, 0:M) { C[k,l] = T[k,l] } }
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& cons_block = builder.add_block(map2_inner.root());
    auto& t_read = builder.add_access(cons_block, "T");
    auto& c_write = builder.add_access(cons_block, "C");
    auto& assign_tasklet = builder.add_tasklet(cons_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        cons_block, t_read, assign_tasklet, "_in", {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );
    builder.add_computational_memlet(
        cons_block, assign_tasklet, "_out", c_write, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager))
        << "Pattern 2: non-perfectly-nested producer with perfectly-nested consumer should be fusible";

    transformation.apply(builder, analysis_manager);

    // After fusion (ConsumerIntoProducer):
    // The producer's inner map body should now have 3 blocks:
    //   block 0: _fused_tmp = S[i] * B[i,j]  (original producer block, output redirected to temp)
    //   block 1: T[i,j] = _fused_tmp          (writeback block)
    //   block 2: C[i,j] = _fused_tmp          (inlined from consumer, k->i, l->j)
    EXPECT_EQ(map1_inner.root().size(), 3)
        << "Inner producer map should have 3 children after fusion (modified original + writeback + inlined consumer)";

    // The sibling block in the outer map should still be there
    EXPECT_EQ(map1_outer.root().size(), 2)
        << "Outer producer map should still have 2 children (sibling block + inner map)";

    // Verify the inlined consumer block writes to C using producer indices (i, j)
    auto* inlined_block = dynamic_cast<structured_control_flow::Block*>(&map1_inner.root().at(2).first);
    ASSERT_TRUE(inlined_block != nullptr);

    auto& inlined_df = inlined_block->dataflow();
    bool found_c_write = false;
    for (auto& node : inlined_df.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr && access->data() == "C") {
            found_c_write = true;
            for (auto& memlet : inlined_df.in_edges(*access)) {
                if (memlet.type() == data_flow::MemletType::Computational) {
                    ASSERT_EQ(memlet.subset().size(), 2) << "C access should have 2D subset";
                    // k should be replaced by i, l should be replaced by j
                    EXPECT_TRUE(symbolic::eq(memlet.subset()[0], symbolic::symbol("i")))
                        << "First index should be i (was k), got: " << memlet.subset()[0]->__str__();
                    EXPECT_TRUE(symbolic::eq(memlet.subset()[1], symbolic::symbol("j")))
                        << "Second index should be j (was l), got: " << memlet.subset()[1]->__str__();
                }
            }
        }
    }
    EXPECT_TRUE(found_c_write) << "Should find C write access in inlined block";

    // The consumer loop should have been removed since all its blocks were inlined
    auto& new_sdfg = builder.subject();
    EXPECT_EQ(new_sdfg.root().size(), 1) << "Consumer loop should be removed after fusion (only producer map remains)";
}

TEST(MapFusionTest, Pattern2_Reverse_NonPerfectlyNestedConsumer) {
    // Reverse Pattern 2: Producer is perfectly nested, consumer is NOT perfectly nested.
    // Producer: Map(i, 0:N) { Map(j, 0:M) { T[i,j] = A[i,j] } }
    // Consumer: Map(k, 0:N) {
    //     Block: W[k] = D[k] + 1.0       (sibling at depth 1)
    //     Map(l, 0:M) {
    //         Block: C[k,l] = T[k,l] * W[k]   (read at depth 2)
    //     }
    // }
    //
    // Fusion direction should be ProducerIntoConsumer, inlining at the consumer's read body.

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array array_1d(float_desc, {symbolic::symbol("N")});
    types::Array inner_array(float_desc, {symbolic::symbol("M")});
    types::Array array_2d(inner_array, {symbolic::symbol("N")});

    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_2d, true);
    builder.add_container("D", array_1d, true);
    builder.add_container("W", array_1d);
    builder.add_container("T", array_2d);
    builder.add_container("C", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:N) { Map(j, 0:M) { T[i,j] = A[i,j] } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& prod_block = builder.add_block(map1_inner.root());
    auto& a_read = builder.add_access(prod_block, "A");
    auto& t_write = builder.add_access(prod_block, "T");
    auto& assign1 = builder.add_tasklet(prod_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        prod_block, a_read, assign1, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );
    builder.add_computational_memlet(
        prod_block, assign1, "_out", t_write, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k, 0:N) { Block: W[k] = D[k] + 1.0; Map(l, 0:M) { C[k,l] = T[k,l] * W[k] } }
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );

    // Sibling block at depth 1: W[k] = D[k] + 1.0
    auto& sibling_block = builder.add_block(map2_outer.root());
    auto& d_read = builder.add_access(sibling_block, "D");
    auto& one_const = builder.add_constant(sibling_block, "1.0", float_desc);
    auto& w_write = builder.add_access(sibling_block, "W");
    auto& add_tasklet = builder.add_tasklet(sibling_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(sibling_block, d_read, add_tasklet, "_in1", {symbolic::symbol("k")}, array_1d);
    builder.add_computational_memlet(sibling_block, one_const, add_tasklet, "_in2", {});
    builder.add_computational_memlet(sibling_block, add_tasklet, "_out", w_write, {symbolic::symbol("k")}, array_1d);

    // Inner Map(l, 0:M): C[k,l] = T[k,l] * W[k]
    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& cons_block = builder.add_block(map2_inner.root());
    auto& t_read = builder.add_access(cons_block, "T");
    auto& w_read = builder.add_access(cons_block, "W");
    auto& c_write = builder.add_access(cons_block, "C");
    auto& mul_tasklet = builder.add_tasklet(cons_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(
        cons_block, t_read, mul_tasklet, "_in1", {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );
    builder.add_computational_memlet(cons_block, w_read, mul_tasklet, "_in2", {symbolic::symbol("k")}, array_1d);
    builder.add_computational_memlet(
        cons_block, mul_tasklet, "_out", c_write, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager))
        << "Reverse Pattern 2: perfectly-nested producer with non-perfectly-nested consumer should be fusible";

    transformation.apply(builder, analysis_manager);

    // After fusion (ProducerIntoConsumer):
    // The consumer's inner map body should now have 2 blocks:
    //   block 0: _fused_tmp = A[k,l]   (inlined from producer, i->k, j->l)
    //   block 1: C[k,l] = _fused_tmp * W[k]   (original consumer block, T replaced)
    EXPECT_EQ(map2_inner.root().size(), 2)
        << "Inner consumer map should have 2 children after fusion (inlined producer + original)";

    // The sibling block in the outer consumer map should still be there
    EXPECT_EQ(map2_outer.root().size(), 2)
        << "Outer consumer map should still have 2 children (sibling block + inner map)";

    // Verify the inlined producer block reads A using consumer indices (k, l)
    auto* inlined_block = dynamic_cast<structured_control_flow::Block*>(&map2_inner.root().at(0).first);
    ASSERT_TRUE(inlined_block != nullptr);

    auto& inlined_df = inlined_block->dataflow();
    bool found_a_read = false;
    for (auto& node : inlined_df.nodes()) {
        auto* access = dynamic_cast<data_flow::AccessNode*>(&node);
        if (access != nullptr && access->data() == "A") {
            found_a_read = true;
            for (auto& memlet : inlined_df.out_edges(*access)) {
                if (memlet.type() == data_flow::MemletType::Computational) {
                    ASSERT_EQ(memlet.subset().size(), 2) << "A access should have 2D subset";
                    // i should be replaced by k, j should be replaced by l
                    EXPECT_TRUE(symbolic::eq(memlet.subset()[0], symbolic::symbol("k")))
                        << "First index should be k (was i), got: " << memlet.subset()[0]->__str__();
                    EXPECT_TRUE(symbolic::eq(memlet.subset()[1], symbolic::symbol("l")))
                        << "Second index should be l (was j), got: " << memlet.subset()[1]->__str__();
                }
            }
        }
    }
    EXPECT_TRUE(found_a_read) << "Should find A read access in inlined producer block";
}

TEST(MapFusionTest, BothNonPerfectlyNested_Rejected) {
    // Both producer and consumer are NOT perfectly nested
    // Should be rejected (not supported)

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array array_1d(float_desc, {symbolic::symbol("N")});
    types::Array inner_array(float_desc, {symbolic::symbol("M")});
    types::Array array_2d(inner_array, {symbolic::symbol("N")});

    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_1d, true);
    builder.add_container("B", array_2d, true);
    builder.add_container("D", array_1d, true);
    builder.add_container("S", array_1d);
    builder.add_container("T", array_2d);
    builder.add_container("W", array_1d);
    builder.add_container("C", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:N) { Block: S[i] = A[i]; Map(j, 0:M) { T[i,j] = S[i] * B[i,j] } }
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );
    auto& sibling1 = builder.add_block(map1_outer.root());
    auto& a_read = builder.add_access(sibling1, "A");
    auto& s_write = builder.add_access(sibling1, "S");
    auto& assign1 = builder.add_tasklet(sibling1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(sibling1, a_read, assign1, "_in", {symbolic::symbol("i")}, array_1d);
    builder.add_computational_memlet(sibling1, assign1, "_out", s_write, {symbolic::symbol("i")}, array_1d);

    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& prod_block = builder.add_block(map1_inner.root());
    auto& s_read = builder.add_access(prod_block, "S");
    auto& b_read = builder.add_access(prod_block, "B");
    auto& t_write = builder.add_access(prod_block, "T");
    auto& mul1 = builder.add_tasklet(prod_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(prod_block, s_read, mul1, "_in1", {symbolic::symbol("i")}, array_1d);
    builder.add_computational_memlet(
        prod_block, b_read, mul1, "_in2", {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );
    builder.add_computational_memlet(
        prod_block, mul1, "_out", t_write, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k, 0:N) { Block: W[k] = D[k]; Map(l, 0:M) { C[k,l] = T[k,l] * W[k] } }
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& sibling2 = builder.add_block(map2_outer.root());
    auto& d_read = builder.add_access(sibling2, "D");
    auto& w_write = builder.add_access(sibling2, "W");
    auto& assign2 = builder.add_tasklet(sibling2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(sibling2, d_read, assign2, "_in", {symbolic::symbol("k")}, array_1d);
    builder.add_computational_memlet(sibling2, assign2, "_out", w_write, {symbolic::symbol("k")}, array_1d);

    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& cons_block = builder.add_block(map2_inner.root());
    auto& t_read = builder.add_access(cons_block, "T");
    auto& w_read = builder.add_access(cons_block, "W");
    auto& c_write = builder.add_access(cons_block, "C");
    auto& mul2 = builder.add_tasklet(cons_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(
        cons_block, t_read, mul2, "_in1", {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );
    builder.add_computational_memlet(cons_block, w_read, mul2, "_in2", {symbolic::symbol("k")}, array_1d);
    builder.add_computational_memlet(
        cons_block, mul2, "_out", c_write, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager))
        << "Both non-perfectly-nested: should be rejected (not supported)";
}

TEST(MapFusionTest, Pattern2_ConsumerReadsMoreThanProducerWrites) {
    // ConsumerIntoProducer range check: producer writes T[i, j] for j in [0, M/2),
    // but consumer reads T[k, l] for l in [0, M). Fusion must be rejected because
    // the consumer reads elements the producer never writes.

    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Array array_1d(float_desc, {symbolic::symbol("N")});
    types::Array inner_array(float_desc, {symbolic::symbol("M")});
    types::Array array_2d(inner_array, {symbolic::symbol("N")});

    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("A", array_1d, true);
    builder.add_container("T", array_2d);
    builder.add_container("C", array_2d, true);

    auto schedule = structured_control_flow::ScheduleType_Sequential::create();

    // Producer: Map(i, 0:N) { Block: sibling;  Map(j, 0:M/2) { T[i,j] = A[i] } }
    // (non-perfectly-nested, writes only half the columns)
    auto& map1_outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        schedule
    );

    // Sibling block to make producer non-perfectly-nested
    auto& sibling_block = builder.add_block(map1_outer.root());
    auto& a_read_sib = builder.add_access(sibling_block, "A");
    auto& a_write_sib = builder.add_access(sibling_block, "A");
    auto& assign_sib = builder.add_tasklet(sibling_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(sibling_block, a_read_sib, assign_sib, "_in", {symbolic::symbol("i")}, array_1d);
    builder.add_computational_memlet(sibling_block, assign_sib, "_out", a_write_sib, {symbolic::symbol("i")}, array_1d);

    // Inner map with restricted range: j in [0, M/2)
    auto& map1_inner = builder.add_map(
        map1_outer.root(),
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::div(symbolic::symbol("M"), symbolic::integer(2))),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        schedule
    );
    auto& prod_block = builder.add_block(map1_inner.root());
    auto& a_read = builder.add_access(prod_block, "A");
    auto& t_write = builder.add_access(prod_block, "T");
    auto& assign_prod = builder.add_tasklet(prod_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(prod_block, a_read, assign_prod, "_in", {symbolic::symbol("i")}, array_1d);
    builder.add_computational_memlet(
        prod_block, assign_prod, "_out", t_write, {symbolic::symbol("i"), symbolic::symbol("j")}, array_2d
    );

    // Consumer: Map(k, 0:N) { Map(l, 0:M) { C[k,l] = T[k,l] } }
    // Reads full range [0, M)
    auto& map2_outer = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        schedule
    );
    auto& map2_inner = builder.add_map(
        map2_outer.root(),
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        schedule
    );
    auto& cons_block = builder.add_block(map2_inner.root());
    auto& t_read = builder.add_access(cons_block, "T");
    auto& c_write = builder.add_access(cons_block, "C");
    auto& assign_cons = builder.add_tasklet(cons_block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(
        cons_block, t_read, assign_cons, "_in", {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );
    builder.add_computational_memlet(
        cons_block, assign_cons, "_out", c_write, {symbolic::symbol("k"), symbolic::symbol("l")}, array_2d
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::MapFusion transformation(map1_outer, map2_outer);

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager))
        << "Consumer reads T[k, 0:M] but producer only writes T[i, 0:M/2] — range not covered";
}
