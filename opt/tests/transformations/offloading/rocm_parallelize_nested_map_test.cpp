#include "sdfg/transformations/offloading/rocm_parallelize_nested_map.h"
#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"

#include "sdfg/targets/rocm/rocm.h"

namespace sdfg::rocm {

TEST(ROCMNestedParallelismTransformation, GridSizeExceedsYZLimit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    auto& indvar = builder.add_container("i", int_desc);
    auto& jndvar = builder.add_container("j", int_desc);
    auto& A_device = builder.add_container("__daisy_hip_A", pointer_type);
    auto& B_host = builder.add_container("B", base_desc);

    ScheduleType rocm_schedule = ScheduleType_ROCM::create();
    ScheduleType_ROCM::dimension(rocm_schedule, ROCMDimension::X);
    ScheduleType_ROCM::block_size(rocm_schedule, symbolic::integer(64));

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& map = builder.add_map(root, symbolic::symbol("i"), condition, init, update, rocm_schedule);

    ScheduleType schedule2 = ScheduleType_Sequential::create();

    // 524288 iterations with block_size=8 -> grid_size=65536, exceeds Y/Z limit of 65535
    auto condition2 = symbolic::Lt(symbolic::symbol("j"), symbolic::integer(524288));
    auto init2 = symbolic::integer(0);
    auto update2 = symbolic::add(symbolic::symbol("j"), symbolic::integer(1));

    auto& map2 = builder.add_map(map.root(), symbolic::symbol("j"), condition2, init2, update2, schedule2);

    auto& block = builder.add_block(map2.root());
    auto& access = builder.add_access(block, "__daisy_hip_A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& constant = builder.add_constant(block, "0.0f", types::Scalar(types::PrimitiveType::Float));

    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, types::Scalar(types::PrimitiveType::Float));
    builder.add_computational_memlet(
        block, tasklet, "out_", access, {symbolic::add(symbolic::symbol("i"), symbolic::symbol("j"))}, pointer_type
    );

    auto& block2 = builder.add_block(root);
    auto& access2 = builder.add_access(block2, "__daisy_hip_A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& access_B = builder.add_access(block2, "B");

    builder.add_computational_memlet(block2, access2, tasklet2, "in_", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block2, tasklet2, "out_", access_B, {}, base_desc);

    transformations::ROCMParallelizeNestedMap transformation(map2, 8);
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(ROCMNestedParallelismTransformation, GridSizeWithinYZLimit) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    auto& indvar = builder.add_container("i", int_desc);
    auto& jndvar = builder.add_container("j", int_desc);
    auto& A_device = builder.add_container("__daisy_hip_A", pointer_type);
    auto& B_host = builder.add_container("B", base_desc);

    ScheduleType rocm_schedule = ScheduleType_ROCM::create();
    ScheduleType_ROCM::dimension(rocm_schedule, ROCMDimension::X);
    ScheduleType_ROCM::block_size(rocm_schedule, symbolic::integer(64));

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& map = builder.add_map(root, symbolic::symbol("i"), condition, init, update, rocm_schedule);

    ScheduleType schedule2 = ScheduleType_Sequential::create();

    // 524280 iterations with block_size=8 -> grid_size=65535, exactly at Y/Z limit
    auto condition2 = symbolic::Lt(symbolic::symbol("j"), symbolic::integer(524280));
    auto init2 = symbolic::integer(0);
    auto update2 = symbolic::add(symbolic::symbol("j"), symbolic::integer(1));

    auto& map2 = builder.add_map(map.root(), symbolic::symbol("j"), condition2, init2, update2, schedule2);

    auto& block = builder.add_block(map2.root());
    auto& access = builder.add_access(block, "__daisy_hip_A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& constant = builder.add_constant(block, "0.0f", types::Scalar(types::PrimitiveType::Float));

    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, types::Scalar(types::PrimitiveType::Float));
    builder.add_computational_memlet(
        block, tasklet, "out_", access, {symbolic::add(symbolic::symbol("i"), symbolic::symbol("j"))}, pointer_type
    );

    auto& block2 = builder.add_block(root);
    auto& access2 = builder.add_access(block2, "__daisy_hip_A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& access_B = builder.add_access(block2, "B");

    builder.add_computational_memlet(block2, access2, tasklet2, "in_", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block2, tasklet2, "out_", access_B, {}, base_desc);

    transformations::ROCMParallelizeNestedMap transformation(map2, 8);
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

} // namespace sdfg::rocm
