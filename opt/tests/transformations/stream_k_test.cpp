#include "sdfg/transformations/stream_k.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace {

// Build the canonical post-offload shape StreamK targets:
//   Map @ <grid_level> (j) { [Reduce{op} @ Z_GRID (k) over C] }
// and return can_be_applied on the grid Map. Knobs flip each precondition.
static bool applies(
    gpu::TargetLevel grid_level,
    structured_control_flow::ReductionOperation op,
    bool include_reduce,
    bool constant_grid_trip,
    bool constant_reduce_trip = true
) {
    builder::StructuredSDFGBuilder builder("streamk", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar i32(types::PrimitiveType::Int32);
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer cptr(f32);
    builder.add_container("j", i32);
    builder.add_container("k", i32);
    builder.add_container("C", cptr, true);
    builder.add_container("A", cptr, true);
    builder.add_container("Nsym", i32, true);

    auto grid_sched =
        gpu::ScheduleType_GPU_Offload::create<cuda::ScheduleType_CUDA_Offload>(grid_level, symbolic::integer(256));
    auto j = symbolic::symbol("j");
    symbolic::Condition jcond = constant_grid_trip ? symbolic::Lt(j, symbolic::integer(16))
                                                   : symbolic::Lt(j, symbolic::symbol("Nsym"));
    auto& gmap =
        builder.add_map(root, j, jcond, symbolic::integer(0), symbolic::add(j, symbolic::integer(1)), grid_sched);

    if (include_reduce) {
        auto red_sched = gpu::ScheduleType_GPU_Offload::create<
            cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::Z_GRID, symbolic::integer(2));
        auto k = symbolic::symbol("k");
        symbolic::Condition kcond = constant_reduce_trip ? symbolic::Lt(k, symbolic::integer(32))
                                                         : symbolic::Lt(k, symbolic::symbol("Nsym"));
        builder.add_reduce(
            gmap.root(),
            k,
            kcond,
            symbolic::integer(0),
            symbolic::add(k, symbolic::integer(1)),
            {structured_control_flow::ReductionInfo{op, "C"}},
            red_sched
        );
    }

    transformations::StreamK xform(gmap);
    analysis::AnalysisManager analysis_manager(builder.subject());
    return xform.can_be_applied(builder, analysis_manager);
}

using ReduceOp = structured_control_flow::ReductionOperation;

// Positive: a grid-level tile band containing an associative (Add) reduction
// with constant tile and panel counts.
TEST(StreamKTest, AcceptsGridAddReduction) {
    EXPECT_TRUE(applies(gpu::TargetLevel::X_GRID, ReduceOp::Add, true, true));
}

// No reduction axis to fold -> a body-agnostic persistent rewrite is not
// Stream-K and buys nothing; reject.
TEST(StreamKTest, RejectsWithoutReduction) {
    EXPECT_FALSE(applies(gpu::TargetLevel::X_GRID, ReduceOp::Add, false, true));
}

// Splitting a non-associative / non-atomic reduction across blocks is illegal.
TEST(StreamKTest, RejectsNonAddReduction) {
    EXPECT_FALSE(applies(gpu::TargetLevel::X_GRID, ReduceOp::Mul, true, true));
    EXPECT_FALSE(applies(gpu::TargetLevel::X_GRID, ReduceOp::Max, true, true));
}

// The anchor must be a grid-mapped band (block-level schedule is not a grid).
TEST(StreamKTest, RejectsNonGridSchedule) {
    EXPECT_FALSE(applies(gpu::TargetLevel::X_BLOCK, ReduceOp::Add, true, true));
}

// Non-constant tile / panel counts make the flat (tile x panel) decode ill-defined.
TEST(StreamKTest, RejectsSymbolicGridTrip) {
    EXPECT_FALSE(applies(gpu::TargetLevel::X_GRID, ReduceOp::Add, true, /*constant_grid_trip=*/false));
}
TEST(StreamKTest, RejectsSymbolicReductionTrip) {
    EXPECT_FALSE(applies(gpu::TargetLevel::X_GRID, ReduceOp::Add, true, true, /*constant_reduce_trip=*/false));
}

// Serialization round-trips the parameters and re-resolves the anchor loop.
TEST(StreamKTest, JsonRoundTrip) {
    builder::StructuredSDFGBuilder builder("streamk", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar i32(types::PrimitiveType::Int32);
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer cptr(f32);
    builder.add_container("j", i32);
    builder.add_container("k", i32);
    builder.add_container("C", cptr, true);

    auto grid_sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(256));
    auto j = symbolic::symbol("j");
    auto& gmap = builder.add_map(
        root,
        j,
        symbolic::Lt(j, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        grid_sched
    );
    auto red_sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::Z_GRID, symbolic::integer(2));
    auto k = symbolic::symbol("k");
    builder.add_reduce(
        gmap.root(),
        k,
        symbolic::Lt(k, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{ReduceOp::Add, "C"}},
        red_sched
    );

    transformations::StreamK xform(gmap, /*num_blocks=*/8);
    nlohmann::json desc;
    xform.to_json(desc);
    EXPECT_EQ(desc["transformation_type"], "StreamK");
    EXPECT_EQ(desc["parameters"]["num_blocks"], 8u);

    auto restored = transformations::StreamK::from_json(builder, desc);
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(restored.can_be_applied(builder, analysis_manager));
}

// apply() rewrites the tile band into a fixed persistent grid whose worker loop
// walks the flat (tile x panel) space, with the reduction moved inside, re-bounded
// to a segment, and forced to merge atomically.
TEST(StreamKTest, ApplyBuildsPersistentWorkerLoop) {
    builder::StructuredSDFGBuilder builder("streamk", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar i32(types::PrimitiveType::Int32);
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer cptr(f32);
    builder.add_container("j", i32);
    builder.add_container("k", i32);
    builder.add_container("C", cptr, true);

    auto grid_sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(256));
    auto j = symbolic::symbol("j");
    auto& gmap = builder.add_map(
        root,
        j,
        symbolic::Lt(j, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        grid_sched
    );
    auto red_sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::Z_GRID, symbolic::integer(2));
    auto k = symbolic::symbol("k");
    builder.add_reduce(
        gmap.root(),
        k,
        symbolic::Lt(k, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{ReduceOp::Add, "C"}},
        red_sched
    );

    transformations::StreamK xform(gmap, /*num_blocks=*/336);
    analysis::AnalysisManager analysis_manager(builder.subject());
    ASSERT_TRUE(xform.can_be_applied(builder, analysis_manager));
    xform.apply(builder, analysis_manager);

    // Grid band is now the persistent block index.
    EXPECT_EQ(gmap.indvar()->get_name(), "__streamk_bid");
    EXPECT_TRUE(symbolic::eq(gpu::ScheduleType_GPU_Offload::parallel_size(gmap.schedule_type()), symbolic::integer(336))
    );

    // Its body is a single sequential worker loop.
    ASSERT_EQ(gmap.root().size(), 1u);
    auto* worker = dynamic_cast<structured_control_flow::For*>(&gmap.root().at(0));
    ASSERT_NE(worker, nullptr);
    EXPECT_EQ(worker->indvar()->get_name(), "__streamk_tile");

    // Worker body: a barrier (WAR across the walk) then the atomic-merge reduce.
    ASSERT_EQ(worker->root().size(), 2u);
    auto* moved = dynamic_cast<structured_control_flow::Reduce*>(&worker->root().at(1));
    ASSERT_NE(moved, nullptr);
    EXPECT_EQ(gpu::ScheduleType_GPU_Offload::partial_storage(moved->schedule_type()), gpu::ReduceStrategy::Global);
}

// apply() on the real pipeline shape -- grid tile band -> block map -> a
// SEQUENTIAL KC-panel reduction below the block maps -- inserts a degenerate
// grid-level (Z_GRID / Global) merge reduce above the block maps (the atomic
// trigger) and re-bounds the panel reduction to the worker segment.
TEST(StreamKTest, ApplyInsertsGridMergeAboveBlockMaps) {
    builder::StructuredSDFGBuilder builder("streamk", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar i32(types::PrimitiveType::Int32);
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer cptr(f32);
    builder.add_container("j", i32);
    builder.add_container("jt", i32);
    builder.add_container("k", i32);
    builder.add_container("C", cptr, true);

    auto grid_sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(256));
    auto j = symbolic::symbol("j");
    auto& gmap = builder.add_map(
        root,
        j,
        symbolic::Lt(j, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        grid_sched
    );

    // Block-level thread map (X_BLOCK), NOT part of the grid tile band.
    auto block_sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(16));
    auto jt = symbolic::symbol("jt");
    auto& bmap = builder.add_map(
        gmap.root(),
        jt,
        symbolic::Lt(jt, symbolic::integer(16)),
        symbolic::integer(0),
        symbolic::add(jt, symbolic::integer(1)),
        block_sched
    );

    // Sequential KC-panel reduction over C, deep below the block map.
    auto k = symbolic::symbol("k");
    builder.add_reduce(
        bmap.root(),
        k,
        symbolic::Lt(k, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{ReduceOp::Add, "C"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );

    transformations::StreamK xform(gmap, /*num_blocks=*/336);
    analysis::AnalysisManager analysis_manager(builder.subject());
    ASSERT_TRUE(xform.can_be_applied(builder, analysis_manager));
    xform.apply(builder, analysis_manager);

    // gmap -> worker(For) -> [barrier, kmerge(Reduce Z_GRID Global)] -> block -> reduce.
    ASSERT_EQ(gmap.root().size(), 1u);
    auto* worker = dynamic_cast<structured_control_flow::For*>(&gmap.root().at(0));
    ASSERT_NE(worker, nullptr);
    ASSERT_EQ(worker->root().size(), 2u);
    auto* kmerge = dynamic_cast<structured_control_flow::Reduce*>(&worker->root().at(1));
    ASSERT_NE(kmerge, nullptr);
    EXPECT_EQ(kmerge->indvar()->get_name(), "__streamk_merge");
    EXPECT_EQ(gpu::ScheduleType_GPU_Offload::partial_storage(kmerge->schedule_type()), gpu::ReduceStrategy::Global);
    EXPECT_EQ(gpu::ScheduleType_GPU_Offload::target_level(kmerge->schedule_type()), gpu::TargetLevel::Z_GRID);

    // The block map is preserved directly under the merge.
    ASSERT_EQ(kmerge->root().size(), 1u);
    auto* moved_block = dynamic_cast<structured_control_flow::Map*>(&kmerge->root().at(0));
    ASSERT_NE(moved_block, nullptr);
    EXPECT_EQ(moved_block->indvar()->get_name(), "jt");

    // The panel reduction is re-bounded to a segment (init is no longer literal 0).
    ASSERT_EQ(moved_block->root().size(), 1u);
    auto* panel = dynamic_cast<structured_control_flow::Reduce*>(&moved_block->root().at(0));
    ASSERT_NE(panel, nullptr);
    EXPECT_FALSE(symbolic::eq(panel->init(), symbolic::integer(0)));
}

} // namespace
