#include "sdfg/transformations/offloading/gpu_offload_nested_loop.h"
#include <gtest/gtest.h>

#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/passes/offloading/reduction_shared_memory_delinearization.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/tiles/analysis/reduction_buffer_analysis.h"

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
    EXPECT_TRUE(
        symbolic::eq(cuda::ScheduleType_CUDA_Offload::parallel_size(block.schedule_type()), symbolic::integer(256))
    );
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

    serializer::JSONSerializer serializer;
    const auto before = serializer.serialize(builder.subject());
    EXPECT_TRUE(analysis_manager.get<tiles::ReductionBufferAnalysis>()
                    .supports_schedule(reduce, gpu_schedule(gpu::TargetLevel::X_BLOCK, 256)));
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    EXPECT_EQ(serializer.serialize(builder.subject()), before);
    transformation.apply(builder, analysis_manager);
    EXPECT_EQ(analysis_manager.get<tiles::ReductionBufferAnalysis>().require(reduce, "__daisy_cuda_A").shared_bytes, 1024);
    passes::ReductionSharedMemoryDelinearization packing;
    ASSERT_TRUE(packing.run_pass(builder, analysis_manager));
    const auto materialized = serializer.serialize(builder.subject());
    Transform resized(reduce, gpu::TargetLevel::X_BLOCK, symbolic::integer(128));
    EXPECT_FALSE(resized.can_be_applied(builder, analysis_manager));
    EXPECT_THROW(resized.apply(builder, analysis_manager), InvalidSDFGException);
    EXPECT_EQ(serializer.serialize(builder.subject()), materialized);
}

// A reduce with warp-sized lanes launches parallel_size * lanes threads, each folding its own
// iteration: the shared partials are sized for every launched thread.
TEST(GPUOffloadNestedLoopTest, ReduceWithLanesApplies) {
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

    analysis::AnalysisManager analysis_manager(builder.subject());
    Transform not_warp_sized(reduce, gpu::TargetLevel::X_BLOCK, symbolic::integer(4), 16);
    EXPECT_FALSE(not_warp_sized.can_be_applied(builder, analysis_manager));

    Transform transformation(reduce, gpu::TargetLevel::X_BLOCK, symbolic::integer(4), 32);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::lanes(reduce.schedule_type())->as_int(), 32);
    EXPECT_EQ(analysis_manager.get<tiles::ReductionBufferAnalysis>().require(reduce, "__daisy_cuda_A").shared_bytes, 512);
}

// Register partials of a lanes reduce live in registers and are combined by shuffles only: a
// nested block reduction of the same accumulator has no shared buffer to fold into.
TEST(GPUOffloadNestedLoopTest, RegisterLanesReduceRejectsNestedBlockReduce) {
    for (auto strategy : {gpu::ReduceStrategy::Register, gpu::ReduceStrategy::Shared}) {
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
        std::vector<structured_control_flow::ReductionInfo> reductions{
            {structured_control_flow::ReductionOperation::Add, "__daisy_cuda_A"}
        };
        auto lanes_schedule = gpu_schedule(gpu::TargetLevel::X_BLOCK, 1);
        gpu::ScheduleType_GPU_Offload::lanes(lanes_schedule, symbolic::integer(32));
        gpu::ScheduleType_GPU_Offload::partial_storage(lanes_schedule, strategy);
        auto& outer = builder.add_reduce(
            grid.root(), symbolic::symbol("j"), lt("j", 64), symbolic::zero(), step("j"), reductions, lanes_schedule
        );
        auto& inner = builder.add_reduce(
            outer.root(),
            symbolic::symbol("k"),
            lt("k", 4),
            symbolic::zero(),
            step("k"),
            reductions,
            structured_control_flow::ScheduleType_Sequential::create()
        );
        add_device_store(builder, inner.root(), "__daisy_cuda_A", symbolic::symbol("i"));

        analysis::AnalysisManager analysis_manager(builder.subject());
        Transform transformation(inner, gpu::TargetLevel::Y_BLOCK, symbolic::integer(4));
        const bool shared = strategy == gpu::ReduceStrategy::Shared;
        EXPECT_EQ(transformation.can_be_applied(builder, analysis_manager), shared);

        builder.update_schedule_type(inner, gpu_schedule(gpu::TargetLevel::Y_BLOCK, 4));
        analysis_manager.invalidate_all();
        auto info = analysis_manager.get<tiles::ReductionBufferAnalysis>().buffer(inner, "__daisy_cuda_A");
        if (shared) {
            EXPECT_EQ(info.status, tiles::ReductionBufferStatus::Exact) << info.diagnostic;
        } else {
            EXPECT_NE(info.status, tiles::ReductionBufferStatus::Exact);
            EXPECT_NE(info.diagnostic.find("Register (lanes)"), std::string::npos) << info.diagnostic;
        }
    }
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

// Y/Z block dimensions may sit directly under an X_GRID (one grid row per block); without an X_GRID
// they still require their own grid dimension.
TEST(GPUOffloadNestedLoopTest, YZBlockDirectlyUnderXGrid) {
    const std::vector<std::tuple<gpu::TargetLevel, gpu::TargetLevel, bool>> cases = {
        {gpu::TargetLevel::X_GRID, gpu::TargetLevel::Y_BLOCK, true},
        {gpu::TargetLevel::X_GRID, gpu::TargetLevel::Z_BLOCK, true},
        {gpu::TargetLevel::Z_GRID, gpu::TargetLevel::Y_BLOCK, false},
        {gpu::TargetLevel::Y_GRID, gpu::TargetLevel::Z_BLOCK, false},
    };
    for (const auto& [grid_level, block_level, applies] : cases) {
        SCOPED_TRACE(static_cast<int>(block_level));
        builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
        types::Scalar int_desc(types::PrimitiveType::Int32);
        builder.add_container("i", int_desc);
        builder.add_container("j", int_desc);
        builder.add_container("__daisy_cuda_A", types::Pointer(types::Scalar(types::PrimitiveType::Float)));

        auto& grid = builder.add_map(
            builder.subject().root(),
            symbolic::symbol("i"),
            lt("i", 128),
            symbolic::zero(),
            step("i"),
            gpu_schedule(grid_level, 128)
        );
        auto& block = builder.add_map(
            grid.root(),
            symbolic::symbol("j"),
            lt("j", 4),
            symbolic::zero(),
            step("j"),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        add_device_store(
            builder,
            block.root(),
            "__daisy_cuda_A",
            symbolic::add(symbolic::mul(symbolic::symbol("i"), symbolic::integer(4)), symbolic::symbol("j"))
        );

        Transform transformation(block, block_level, symbolic::integer(4));
        analysis::AnalysisManager analysis_manager(builder.subject());
        ASSERT_EQ(transformation.can_be_applied(builder, analysis_manager), applies);
        if (applies) {
            transformation.apply(builder, analysis_manager);
            EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(block.schedule_type()), block_level);
        }
    }
}

// Below an X_GRID -> Y_BLOCK nest, a grid level (under a block) or a level already used by an
// ancestor is rejected; a distinct block level still applies.
TEST(GPUOffloadNestedLoopTest, GridUnderBlockOrRepeatedLevelRejected) {
    const std::vector<std::pair<gpu::TargetLevel, bool>> cases = {
        {gpu::TargetLevel::Y_GRID, false},
        {gpu::TargetLevel::Z_GRID, false},
        {gpu::TargetLevel::Y_BLOCK, false},
        {gpu::TargetLevel::X_BLOCK, true},
    };
    for (const auto& [inner_level, applies] : cases) {
        SCOPED_TRACE(static_cast<int>(inner_level));
        builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
        types::Scalar int_desc(types::PrimitiveType::Int32);
        builder.add_container("i", int_desc);
        builder.add_container("j", int_desc);
        builder.add_container("k", int_desc);
        builder.add_container("__daisy_cuda_A", types::Pointer(types::Scalar(types::PrimitiveType::Float)));

        auto& grid = builder.add_map(
            builder.subject().root(),
            symbolic::symbol("i"),
            lt("i", 128),
            symbolic::zero(),
            step("i"),
            gpu_schedule(gpu::TargetLevel::X_GRID, 128)
        );
        auto& block = builder.add_map(
            grid.root(),
            symbolic::symbol("j"),
            lt("j", 4),
            symbolic::zero(),
            step("j"),
            gpu_schedule(gpu::TargetLevel::Y_BLOCK, 4)
        );
        auto& inner = builder.add_map(
            block.root(),
            symbolic::symbol("k"),
            lt("k", 4),
            symbolic::zero(),
            step("k"),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto index = symbolic::
            add(symbolic::mul(symbolic::symbol("i"), symbolic::integer(16)),
                symbolic::add(symbolic::mul(symbolic::symbol("j"), symbolic::integer(4)), symbolic::symbol("k")));
        add_device_store(builder, inner.root(), "__daisy_cuda_A", index);

        Transform transformation(inner, inner_level, symbolic::integer(4));
        analysis::AnalysisManager analysis_manager(builder.subject());
        EXPECT_EQ(transformation.can_be_applied(builder, analysis_manager), applies);
    }
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

namespace {

// Grid map over rows with a sequential reduce carrying the online-softmax pair (m, d).
bool can_offload_rescale_group(gpu::TargetLevel level, int64_t parallel_size) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar int_desc(types::PrimitiveType::Int32);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    builder.add_container("i", int_desc);
    builder.add_container("k", int_desc);
    builder.add_container("__daisy_cuda_m", pointer_type);
    builder.add_container("__daisy_cuda_d", pointer_type);

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        gpu_schedule(gpu::TargetLevel::X_GRID, 128)
    );
    auto& reduce = builder.add_reduce(
        grid.root(),
        symbolic::symbol("k"),
        lt("k", 1024),
        symbolic::zero(),
        step("k"),
        {{.operation = structured_control_flow::ReductionOperation::Max, .container = "__daisy_cuda_m"},
         {.operation = structured_control_flow::ReductionOperation::Add,
          .container = "__daisy_cuda_d",
          .rescale = "__daisy_cuda_m"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_device_store(builder, reduce.root(), "__daisy_cuda_m", symbolic::symbol("i"));
    add_device_store(builder, reduce.root(), "__daisy_cuda_d", symbolic::symbol("i"));

    analysis::AnalysisManager analysis_manager(builder.subject());
    Transform transformation(reduce, level, symbolic::integer(parallel_size));
    return transformation.can_be_applied(builder, analysis_manager);
}

} // namespace

TEST(GPUOffloadNestedLoopTest, RescaleGroup_XBlockWarpMultipleApplies) {
    EXPECT_TRUE(can_offload_rescale_group(gpu::TargetLevel::X_BLOCK, 256));
}

TEST(GPUOffloadNestedLoopTest, RescaleGroup_NonWarpTilingRejected) {
    EXPECT_FALSE(can_offload_rescale_group(gpu::TargetLevel::X_BLOCK, 48));
}

} // namespace sdfg
