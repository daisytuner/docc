#include "sdfg/transformations/offloading/rocm_transform.h"

#include <gtest/gtest.h>

using namespace sdfg;

// A sequential map is offloadable to a ROCM kernel.
TEST(ROCMTransformTest, MapWithSequentialSchedule) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    builder.add_container("i", int_desc);

    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    rocm::ROCMTransform transformation(map, 64);
    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);

    EXPECT_EQ(map.schedule_type().value(), rocm::ScheduleType_ROCM::value());
}

// Regression test (remote-tuning double-offload): re-applying ROCMTransform to a map that
// already carries a ROCM schedule must be rejected, so tuned cutouts are not offloaded twice.
TEST(ROCMTransformTest, RejectsAlreadyROCMScheduledMap) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    builder.add_container("i", int_desc);

    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        rocm::ScheduleType_ROCM::create()
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    rocm::ROCMTransform transformation(map, 128);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));

    // Schedule left untouched.
    EXPECT_EQ(map.schedule_type().value(), rocm::ScheduleType_ROCM::value());
}
