#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"

#include "printf_data_offloading_node.h"
#include "printf_target.h"
#include "printf_transform.h"

namespace sdfg::printf_target {

TEST(PrintfTransformTest, MapWithSequentialSchedule) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

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

    // Define loop with sequential schedule
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");
    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(indvar, bound);
    auto update = symbolic::add(indvar, symbolic::integer(1));

    auto& loop =
        builder
            .add_map(root, indvar, condition, init, update, structured_control_flow::ScheduleType_Sequential::create());
    auto& body = loop.root();

    // Add computation: A[i] = A[i]
    auto& block = builder.add_block(body);
    auto& A_in = builder.add_access(block, "A");
    auto& A_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, A_in, tasklet, "_in", {indvar}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", A_out, {indvar}, desc);

    // Apply transformation
    analysis::AnalysisManager analysis_manager(builder.subject());
    PrintfTransform transformation(loop);

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    // Verify schedule type changed to Printf
    EXPECT_EQ(loop.schedule_type().value(), ScheduleType_Printf::value());
}

TEST(PrintfTransformTest, TransformNameTest) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(loop.root());

    PrintfTransform transformation(loop);
    EXPECT_EQ(transformation.name(), "PrintfTransform");
}

TEST(PrintfTransformTest, SerializationTest) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(loop.root());

    size_t map_id = loop.element_id();

    PrintfTransform transformation(loop);

    // Test to_json
    nlohmann::json j;
    EXPECT_NO_THROW(transformation.to_json(j));

    // Verify JSON structure
    EXPECT_EQ(j["transformation_type"], "PrintfTransform");
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], map_id);
    EXPECT_EQ(j["parameters"]["allow_dynamic_sizes"], false);
}

TEST(PrintfTransformTest, DeserializationTest) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(loop.root());

    size_t map_id = loop.element_id();

    // Create JSON for deserialization
    nlohmann::json j;
    j["transformation_type"] = "PrintfTransform";
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    j["subgraph"]["0"]["element_id"] = map_id;
    j["subgraph"]["0"]["type"] = "map";
    j["parameters"] = nlohmann::json::object();
    j["parameters"]["allow_dynamic_sizes"] = false;

    // Test from_json
    PrintfTransform deserialized = PrintfTransform::from_json(builder, j);
    EXPECT_EQ(deserialized.name(), "PrintfTransform");
}

TEST(PrintfTransformTest, CopyPrefixTest) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    builder.add_block(loop.root());

    // The copy prefix should be the PRINTF_DEVICE_PREFIX
    EXPECT_EQ(PRINTF_DEVICE_PREFIX, "__printf_device_");
}

TEST(PrintfTransformTest, TargetTypeTest) {
    // Verify the target type constant
    EXPECT_EQ(TargetType_Printf.value(), "Printf");
}

} // namespace sdfg::printf_target
