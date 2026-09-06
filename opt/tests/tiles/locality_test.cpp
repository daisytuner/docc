#include "sdfg/tiles/locality.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/tiles/tile.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace {

/// A cooperative/per-thread scratchpad (GPU-like) axis at @p level for required_space tests.
tiles::TileAxis make_gpu_axis(bool cooperative, tiles::Level level = tiles::Level::Group) {
    return tiles::TileAxis(
        symbolic::symbol("i"),
        cooperative ? tiles::Role::Cooperative : tiles::Role::Private,
        tiles::AxisSchedule(
            level, tiles::default_space(level), /*has_scratchpad=*/true, /*spatial_axis=*/0, symbolic::integer(32)
        )
    );
}

/// A cooperative/per-thread host (global-only) parallel axis for required_space tests.
tiles::TileAxis make_cpu_axis(bool cooperative) {
    return tiles::TileAxis(
        symbolic::symbol("i"),
        cooperative ? tiles::Role::Cooperative : tiles::Role::Private,
        tiles::AxisSchedule(tiles::Level::Device, tiles::Space::Global, /*has_scratchpad=*/false)
    );
}

/// Analyze the placement at @p loop for a tile with the given @p bases.
tiles::LocalityPlan plan_for(
    structured_control_flow::StructuredLoop& loop, const symbolic::MultiExpression& bases, analysis::AnalysisManager& am
) {
    return tiles::LocalityPlan::analyze(loop, tiles::TileAxis::enclosing(loop, bases), am);
}

} // namespace

// =====================================================================
// LocalityPlan::analyze: schedule classification
// =====================================================================

// Sequential For nest: no parallel axes, not inside a kernel.
TEST(LocalityTest, Plan_Sequential_NoParallelDims) {
    builder::StructuredSDFGBuilder builder("plan_seq", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto K = symbolic::symbol("K");
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto& loop_i =
        builder.add_for(seq, i, symbolic::Lt(i, K), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)));
    auto& loop_k =
        builder
            .add_for(loop_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    auto plan = plan_for(loop_k, {k}, am);

    EXPECT_TRUE(plan.axes().empty());
    EXPECT_FALSE(plan.inside_scratchpad_scope());
    EXPECT_FALSE(plan.has_scratchpad_cooperative());
    EXPECT_FALSE(plan.loop_is_outermost());
    EXPECT_EQ(plan.required_space(/*written*/ false), tiles::Space::Register);
    EXPECT_EQ(plan.required_space(/*written*/ true), tiles::Space::Register);
}

// GPU map whose indvar appears in the tile base → per-thread axis.
TEST(LocalityTest, Plan_GpuPerThread) {
    builder::StructuredSDFGBuilder builder("plan_gpu_perthread", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_i =
        builder
            .add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched_x);
    auto& loop_k =
        builder
            .add_for(map_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    auto plan = plan_for(loop_k, {symbolic::add(symbolic::mul(i, K), k)}, am); // base depends on i → per-thread

    ASSERT_EQ(plan.axes().size(), 1u);
    EXPECT_TRUE(plan.axes()[0].schedule().has_scratchpad());
    EXPECT_FALSE(plan.axes()[0].cooperative());
    EXPECT_TRUE(plan.inside_scratchpad_scope());
    EXPECT_FALSE(plan.has_scratchpad_cooperative());
    // Per-thread read or write both localize to a private buffer.
    EXPECT_EQ(plan.required_space(false), tiles::Space::Register);
    EXPECT_EQ(plan.required_space(true), tiles::Space::Register);
}

// GPU map whose indvar is absent from the tile base → cooperative axis.
TEST(LocalityTest, Plan_GpuCooperative) {
    builder::StructuredSDFGBuilder builder("plan_gpu_coop", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched_x = cuda::ScheduleType_CUDA::create();
    gpu::gpu_block_size(sched_x, symbolic::integer(32));
    auto& map_i =
        builder
            .add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched_x);
    auto& loop_k =
        builder
            .add_for(map_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    auto plan = plan_for(loop_k, {k}, am); // base independent of i → all threads share the tile

    ASSERT_EQ(plan.axes().size(), 1u);
    EXPECT_TRUE(plan.axes()[0].schedule().has_scratchpad());
    EXPECT_TRUE(plan.axes()[0].cooperative());
    EXPECT_TRUE(plan.inside_scratchpad_scope());
    EXPECT_TRUE(plan.has_scratchpad_cooperative());
    EXPECT_FALSE(plan.loop_is_outermost());
    // Cooperative read → shared; cooperative write → reduction we can't lower.
    EXPECT_EQ(plan.required_space(/*written*/ false), tiles::Space::Shared);
    EXPECT_FALSE(plan.required_space(/*written*/ true));
}

// =====================================================================
// LocalityPlan::required_space: storage derivation from a classification
// =====================================================================

TEST(LocalityTest, Derive_Empty_Private) {
    tiles::LocalityPlan plan;
    EXPECT_EQ(plan.required_space(false), tiles::Space::Register);
    EXPECT_EQ(plan.required_space(true), tiles::Space::Register);
}

// A cooperative CPU-parallel axis (tile invariant across threads, e.g. shared ~B):
// a read-only tile replicates per-thread (Register); a written one is a cross-thread
// reduction/race and declines.
TEST(LocalityTest, Derive_CpuCooperative_ReadReplicates_WriteRejects) {
    tiles::LocalityPlan plan({make_cpu_axis(/*cooperative*/ true)});
    EXPECT_EQ(plan.required_space(false), tiles::Space::Register);
    EXPECT_FALSE(plan.required_space(true));
}

TEST(LocalityTest, Derive_CpuPerThread_Private) {
    tiles::LocalityPlan plan({make_cpu_axis(/*cooperative*/ false)});
    EXPECT_EQ(plan.required_space(false), tiles::Space::Register);
    EXPECT_EQ(plan.required_space(true), tiles::Space::Register);
}

TEST(LocalityTest, Derive_GpuCooperativeRead_Shared) {
    tiles::LocalityPlan plan({make_gpu_axis(/*cooperative*/ true)});
    EXPECT_EQ(plan.required_space(/*written*/ false), tiles::Space::Shared);
}

TEST(LocalityTest, Derive_GpuCooperativeWrite_Reject) {
    tiles::LocalityPlan plan({make_gpu_axis(/*cooperative*/ true)});
    EXPECT_FALSE(plan.required_space(/*written*/ true));
}

TEST(LocalityTest, Derive_GpuCooperativeOutermost_Reject) {
    // a shared buffer can't straddle the kernel boundary
    tiles::LocalityPlan plan({make_gpu_axis(/*cooperative*/ true)}, /*loop_is_outermost=*/true);
    EXPECT_FALSE(plan.required_space(false));
}

TEST(LocalityTest, Derive_GpuPerThread_Private) {
    tiles::LocalityPlan plan({make_gpu_axis(/*cooperative*/ false)});
    EXPECT_EQ(plan.required_space(false), tiles::Space::Register);
    EXPECT_EQ(plan.required_space(true), tiles::Space::Register);
}

// A host-level loop that itself is scratchpad-scheduled or wraps a scratchpad kernel
// is not a localization site for a private stack buffer.
TEST(LocalityTest, Derive_HostWrapsGpuKernel_Reject) {
    tiles::LocalityPlan
        plan({}, /*loop_is_outermost=*/false, /*loop_has_scratchpad=*/false, /*has_scratchpad_descendant=*/true);
    EXPECT_FALSE(plan.required_space(false));

    tiles::LocalityPlan plan2({}, /*loop_is_outermost=*/false, /*loop_has_scratchpad=*/true);
    EXPECT_FALSE(plan2.required_space(false));
}

// Cooperation across blocks (grid level) needs grid-wide global memory.
TEST(LocalityTest, Derive_GpuGridCooperativeRead_Global) {
    tiles::LocalityPlan plan({make_gpu_axis(/*cooperative*/ true, tiles::Level::Device)});
    EXPECT_EQ(plan.required_space(/*written*/ false), tiles::Space::Global);
    EXPECT_FALSE(plan.required_space(/*written*/ true));
}

// A read tile with block cooperation lives in shared memory even when it is also
// grid-cooperative: each block redundantly stages its own copy (grid cooperation
// is replication, not a shared buffer). This is the 2D-block GEMM shape.
TEST(LocalityTest, Derive_GpuGridAndBlockCooperativeRead_Shared) {
    tiles::LocalityPlan plan(
        {make_gpu_axis(/*cooperative*/ true, tiles::Level::Group),
         make_gpu_axis(/*cooperative*/ true, tiles::Level::Device)}
    );
    EXPECT_EQ(plan.required_space(/*written*/ false), tiles::Space::Shared);
}

// Warp-only cooperation is served by shuffles, not a staged buffer → Reject.
TEST(LocalityTest, Derive_GpuWarpCooperativeRead_Reject) {
    tiles::LocalityPlan plan({make_gpu_axis(/*cooperative*/ true, tiles::Level::Subgroup)});
    EXPECT_FALSE(plan.required_space(/*written*/ false));
}

// A per-thread warp axis (indvar in the tile base) is fine → private register tile.
TEST(LocalityTest, Derive_GpuWarpPerThread_Private) {
    tiles::LocalityPlan plan({make_gpu_axis(/*cooperative*/ false, tiles::Level::Subgroup)});
    EXPECT_EQ(plan.required_space(false), tiles::Space::Register);
}

// analyze reads the new *_Offload schedule: block-level target, parallel size, sync.
TEST(LocalityTest, Plan_GpuOffload_BlockLevel) {
    builder::StructuredSDFGBuilder builder("plan_offload_block", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(64));
    auto& map_i =
        builder.add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched);
    auto& loop_k =
        builder
            .add_for(map_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    auto plan = plan_for(loop_k, {k}, am); // base independent of i → cooperative across the block

    ASSERT_EQ(plan.axes().size(), 1u);
    EXPECT_TRUE(plan.axes()[0].schedule().has_scratchpad());
    EXPECT_TRUE(plan.axes()[0].cooperative());
    EXPECT_EQ(plan.axes()[0].schedule().level(), tiles::Level::Group);
    EXPECT_TRUE(symbolic::eq(plan.axes()[0].schedule().parallel_size(), symbolic::integer(64)));
    EXPECT_FALSE(plan.axes()[0].schedule().needs_sync());
    EXPECT_EQ(plan.required_space(/*written*/ false), tiles::Space::Shared);
}

// A grid-level *_Offload cooperative read derives to grid-wide global memory.
TEST(LocalityTest, Plan_GpuOffload_GridLevel) {
    builder::StructuredSDFGBuilder builder("plan_offload_grid", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);

    auto sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(128));
    auto& map_i =
        builder.add_map(seq, i, symbolic::Lt(i, N), symbolic::integer(0), symbolic::add(i, symbolic::integer(1)), sched);
    auto& loop_k =
        builder
            .add_for(map_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    auto plan = plan_for(loop_k, {k}, am);

    ASSERT_EQ(plan.axes().size(), 1u);
    EXPECT_EQ(plan.axes()[0].schedule().level(), tiles::Level::Device);
    EXPECT_EQ(plan.required_space(/*written*/ false), tiles::Space::Global);
}

// A GPU-scheduled Reduce enclosing the loop is a cooperative block level too, so
// analyze classifies a read tile inside a block reduction as shared (previously it
// saw only Maps and mis-derived a private per-thread buffer).
TEST(LocalityTest, Plan_GpuOffloadReduce_BlockLevel) {
    builder::StructuredSDFGBuilder builder("plan_offload_reduce_block", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);

    auto sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(64));
    auto& reduce_i = builder.add_reduce(
        seq,
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        sched
    );
    auto& loop_k =
        builder
            .add_for(reduce_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    auto plan = plan_for(loop_k, {k}, am); // base independent of i → cooperative across the block reduce

    ASSERT_EQ(plan.axes().size(), 1u);
    EXPECT_TRUE(plan.axes()[0].schedule().has_scratchpad());
    EXPECT_TRUE(plan.axes()[0].cooperative());
    EXPECT_EQ(plan.axes()[0].schedule().level(), tiles::Level::Group);
    EXPECT_TRUE(symbolic::eq(plan.axes()[0].schedule().parallel_size(), symbolic::integer(64)));
    EXPECT_EQ(plan.required_space(/*written*/ false), tiles::Space::Shared);
}
