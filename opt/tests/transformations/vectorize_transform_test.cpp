#include "sdfg/transformations/vectorize_transform.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(VectorizeTransformTest, MapWithSequentialSchedule) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 = builder.add_map(
        root, indvar1, condition1, init1, update1, structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body1 = loop1.root();

    // Add computation
    // A[j] = A[j]
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar1}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::VectorizeTransform transformation(loop1);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    EXPECT_EQ(loop1.schedule_type().value(), vectorize::ScheduleType_Vectorize::value());
}

TEST(VectorizeTransformTest, MapWithVectorizeSchedule) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop
    auto bound1 = symbolic::symbol("N");
    auto indvar1 = symbolic::symbol("i");
    auto init1 = symbolic::integer(0);
    auto condition1 = symbolic::Lt(indvar1, bound1);
    auto update1 = symbolic::add(indvar1, symbolic::integer(1));

    auto& loop1 =
        builder.add_map(root, indvar1, condition1, init1, update1, vectorize::ScheduleType_Vectorize::create());
    auto& body1 = loop1.root();

    // Add computation
    // A[j] = A[j]
    auto& block = builder.add_block(body1);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar1}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar1}, desc);

    // Apply
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::VectorizeTransform transformation(loop1);
    ASSERT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}


TEST(VectorizeTransformTest, Serialization) {
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
    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    size_t map_id = loop.element_id();

    transformations::VectorizeTransform transformation(loop);

    // Test to_json
    nlohmann::json j;
    EXPECT_NO_THROW(transformation.to_json(j));

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "VectorizeTransform");
    EXPECT_TRUE(j.contains("subgraph"));
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], map_id);
    EXPECT_EQ(j["subgraph"]["0"]["type"], "map");
}

TEST(VectorizeTransformTest, Deserialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    // Define loop
    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    size_t map_id = loop.element_id();

    // Create JSON description
    nlohmann::json j;
    j["transformation_type"] = "VectorizeTransform";
    j["subgraph"] = {{"0", {{"element_id", map_id}, {"type", "map"}}}};

    // Test from_json
    EXPECT_NO_THROW({
        auto deserialized = transformations::VectorizeTransform::from_json(builder, j);
        EXPECT_EQ(deserialized.name(), "VectorizeTransform");
    });

    EXPECT_THROW(
        {
            nlohmann::json invalid_j = j;
            invalid_j["subgraph"]["0"]["element_id"] = 9999; // Non-existent ID
            transformations::VectorizeTransform::from_json(builder, invalid_j);
        },
        transformations::InvalidTransformationDescriptionException
    );
}
