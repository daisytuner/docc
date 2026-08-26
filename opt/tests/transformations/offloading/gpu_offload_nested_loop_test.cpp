#include "sdfg/transformations/offloading/gpu_offload_nested_loop.h"
#include <gtest/gtest.h>

#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg {

using Transform = transformations::GPUOffloadNestedLoop<cuda::ScheduleType_CUDA_Offload>;

namespace {

// Build a GPU schedule for the given target level / parallel size.
structured_control_flow::ScheduleType gpu_schedule(gpu::TargetLevel level, int64_t parallel_size) {
    return cuda::ScheduleType_CUDA_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(level, symbolic::integer(parallel_size));
}

// A simple loop condition `<indvar> < bound`.
symbolic::Condition lt(const std::string& indvar, int64_t bound) {
    return symbolic::Lt(symbolic::symbol(indvar), symbolic::integer(bound));
}

symbolic::Expression step(const std::string& indvar) {
    return symbolic::add(symbolic::symbol(indvar), symbolic::integer(1));
}

// Store 0.0f into `container[index]` so the loop body has a well-defined (race-free) write.
void add_device_store(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::string& container,
    const symbolic::Expression& index
) {
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    auto& block = builder.add_block(parent);
    auto& access = builder.add_access(block, container);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& constant = builder.add_constant(block, "0.0f", base_desc);
    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, base_desc);
    builder.add_computational_memlet(block, tasklet, "out_", access, {index}, pointer_type);
}

} // namespace

// -----------------------------------------------------------------------------
// Positive cases
// -----------------------------------------------------------------------------

TEST(GPUOffloadNestedLoopTest, XBlockNestedInXGridApplies) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(((types::Scalar(types::PrimitiveType::Float))));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& block = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 256),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(
        builder,
        block.root(),
        "__daisy_cuda_A",
        symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::integer(256)), symbolic::symbol("j"))
    );

    Transform transformation(block, gpu::TargetLevel::X_BLOCK, symbolic::integer(256));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));

    transformation.apply(builder, analysis_manager);
    EXPECT_EQ(block.schedule_type().value(), cuda::ScheduleType_CUDA_Offload::value());
    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(block.schedule_type()), gpu::TargetLevel::X_BLOCK);
    EXPECT_TRUE(symbolic::eq(cuda::ScheduleType_CUDA_Offload::parallel_size(block.schedule_type()), symbolic::integer(256))
    );
}

TEST(GPUOffloadNestedLoopTest, WarpNestedInXBlockApplies) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("k", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& xblock = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 8),
        symbolic::zero(),
        step("j"),
        gpu_schedule(gpu::TargetLevel::X_BLOCK, 8)
    );
    auto& warp = builder.add_map(
        xblock.root(),
        symbolic::symbol("k"),
        lt("k", 32),
        symbolic::zero(),
        step("k"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, warp.root(), "__daisy_cuda_A", symbolic::symbol("k"));

    Transform transformation(warp, gpu::TargetLevel::WARP, symbolic::integer(32));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));

    transformation.apply(builder, analysis_manager);
    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(warp.schedule_type()), gpu::TargetLevel::WARP);
}

TEST(GPUOffloadNestedLoopTest, ReduceWithSupportedOperationApplies) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );

    std::vector<structured_control_flow::ReductionInfo> reductions{
        {structured_control_flow::ReductionOperation::Add, "__daisy_cuda_A"}
    };
    auto& reduce = builder.add_reduce(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 256),
        symbolic::zero(),
        step("j"),
        reductions,
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, reduce.root(), "__daisy_cuda_A", symbolic::symbol("i"));

    Transform transformation(reduce, gpu::TargetLevel::X_BLOCK, symbolic::integer(256));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

// -----------------------------------------------------------------------------
// Negative cases
// -----------------------------------------------------------------------------

TEST(GPUOffloadNestedLoopTest, NoParentLoopRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, outer.root(), "__daisy_cuda_A", symbolic::symbol("i"));

    Transform transformation(outer, gpu::TargetLevel::X_BLOCK, symbolic::integer(256));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(GPUOffloadNestedLoopTest, NonMapNonReduceRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& for_loop = builder.add_for(grid.root(), symbolic::symbol("j"), lt("j", 256), symbolic::zero(), step("j"));
    add_device_store(builder, for_loop.root(), "__daisy_cuda_A", symbolic::symbol("j"));

    Transform transformation(for_loop, gpu::TargetLevel::X_BLOCK, symbolic::integer(256));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(GPUOffloadNestedLoopTest, ParallelSizeExceedsBlockLimitRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& block = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 2048),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, block.root(), "__daisy_cuda_A", symbolic::symbol("j"));

    // 2048 > 1024 (X_BLOCK limit)
    Transform transformation(block, gpu::TargetLevel::X_BLOCK, symbolic::integer(2048));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(GPUOffloadNestedLoopTest, NonPositiveParallelSizeRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& block = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 256),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, block.root(), "__daisy_cuda_A", symbolic::symbol("j"));

    Transform transformation(block, gpu::TargetLevel::X_BLOCK, symbolic::integer(0));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(GPUOffloadNestedLoopTest, WarpWrongSizeRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("k", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& xblock = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 8),
        symbolic::zero(),
        step("j"),
        gpu_schedule(gpu::TargetLevel::X_BLOCK, 8)
    );
    auto& warp = builder.add_map(
        xblock.root(),
        symbolic::symbol("k"),
        lt("k", 32),
        symbolic::zero(),
        step("k"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, warp.root(), "__daisy_cuda_A", symbolic::symbol("k"));

    // WARP must be exactly 32.
    Transform transformation(warp, gpu::TargetLevel::WARP, symbolic::integer(16));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(GPUOffloadNestedLoopTest, NoGpuAncestorRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block = builder.add_map(
        outer.root(),
        symbolic::symbol("j"),
        lt("j", 256),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, block.root(), "__daisy_cuda_A", symbolic::symbol("j"));

    Transform transformation(block, gpu::TargetLevel::X_BLOCK, symbolic::integer(256));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(GPUOffloadNestedLoopTest, BlockNotNestedInMatchingGridRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    // X_BLOCK requires an X_GRID ancestor, but only Y_GRID is present.
    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::Y_GRID, 128)
    );
    auto& block = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 256),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, block.root(), "__daisy_cuda_A", symbolic::symbol("j"));

    Transform transformation(block, gpu::TargetLevel::X_BLOCK, symbolic::integer(256));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(GPUOffloadNestedLoopTest, WarpNotNestedInXBlockRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    // WARP requires an X_BLOCK ancestor, but only X_GRID is present.
    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& warp = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 32),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, warp.root(), "__daisy_cuda_A", symbolic::symbol("j"));

    Transform transformation(warp, gpu::TargetLevel::WARP, symbolic::integer(32));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(GPUOffloadNestedLoopTest, DimensionNestedWithinWarpRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("k", int_desc);
    builder.add_container("l", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& xblock = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 8),
        symbolic::zero(),
        step("j"),
        gpu_schedule(gpu::TargetLevel::X_BLOCK, 8)
    );
    auto& warp = builder.add_map(
        xblock.root(),
        symbolic::symbol("k"),
        lt("k", 32),
        symbolic::zero(),
        step("k"),
        gpu_schedule(gpu::TargetLevel::WARP, 32)
    );
    auto& inner = builder.add_map(
        warp.root(),
        symbolic::symbol("l"),
        lt("l", 4),
        symbolic::zero(),
        step("l"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, inner.root(), "__daisy_cuda_A", symbolic::symbol("l"));

    // Nothing may be nested within a WARP.
    Transform transformation(inner, gpu::TargetLevel::X_BLOCK, symbolic::integer(4));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(GPUOffloadNestedLoopTest, BlockProductExceedsLimitRejected) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container("i", int_desc);
    builder.add_container("j", int_desc);
    builder.add_container("k", int_desc);
    builder.add_container("l", int_desc);
    builder.add_container("__daisy_cuda_A", pointer_type);

    auto& xgrid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& ygrid = builder.add_map(
        xgrid.root(),
        symbolic::symbol("j"),
        lt("j", 128),
        symbolic::zero(),
        step("j"),
        gpu_schedule(gpu::TargetLevel::Y_GRID, 128)
    );
    auto& xblock = builder.add_map(
        ygrid.root(),
        symbolic::symbol("k"),
        lt("k", 64),
        symbolic::zero(),
        step("k"),
        gpu_schedule(gpu::TargetLevel::X_BLOCK, 64)
    );
    auto& yblock = builder.add_map(
        xblock.root(),
        symbolic::symbol("l"),
        lt("l", 64),
        symbolic::zero(),
        step("l"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, yblock.root(), "__daisy_cuda_A", symbolic::symbol("l"));

    // 64 (X_BLOCK) * 64 (Y_BLOCK) = 4096 > 1024.
    Transform transformation(yblock, gpu::TargetLevel::Y_BLOCK, symbolic::integer(64));
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

} // namespace sdfg
