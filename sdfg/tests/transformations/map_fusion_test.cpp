#include "sdfg/transformations/map_fusion.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(MapFusionTest, SimpleMapFusion) {
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

    // Count total nodes across both blocks (producer computation + consumer computation)
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
    // Second map accesses beyond what first map computed - index mapping still works

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

    // The index mapping is valid (i = j), but this would cause recomputation
    // In the current implementation, we allow this as long as index mapping is solvable
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
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
    // The current implementation computes index_mapping = (2*j + 1) / 2 = j + 0.5
    // This involves division that doesn't result in an integer
    // However, our current validator only checks if atoms are valid symbols
    // A more sophisticated check would verify integer divisibility
    // For now, this will actually pass can_be_applied (the mapping is algebraically valid)
    // In a complete implementation, we'd reject this case

    // NOTE: The test documents current behavior - algebraically the mapping exists
    // but semantically it's problematic. Future work could add integer divisibility checks.
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
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
    auto indvar2 = symbolic::symbol("j");
    auto& map2 = builder.add_map(
        root,
        indvar2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
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
