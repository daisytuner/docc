#include "sdfg/tiles/transformations/tile_streaming_copy.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"

using namespace sdfg;
using transformations::TileStreamingCopy;

namespace {

// A GPU map over @p indvar < @p size at @p level in @p parent.
structured_control_flow::Map& add_gpu_map(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::string& indvar,
    gpu::TargetLevel level,
    long long size,
    long long lanes = 1
) {
    builder.add_container(indvar, types::Scalar(types::PrimitiveType::Int32));
    auto symbol = symbolic::symbol(indvar);
    auto sched = gpu::ScheduleType_GPU_Offload::create<cuda::ScheduleType_CUDA_Offload>(level, symbolic::integer(size));
    if (lanes != 1) {
        gpu::ScheduleType_GPU_Offload::lanes(sched, symbolic::integer(lanes));
    }
    return builder.add_map(
        parent,
        symbol,
        symbolic::Lt(symbol, symbolic::integer(size)),
        symbolic::zero(),
        symbolic::add(symbol, symbolic::one()),
        sched
    );
}

// Loads 4 floats of A at @p src_offset into a register tile as one 16-byte per-thread VectorSync
// copy-in placed in @p parent.
tiles::TileCopyNode& add_copy_in(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const symbolic::Expression& src_offset
) {
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("A", ptr, true);
    builder.add_container("buf", types::Array(elem, symbolic::integer(4)));
    auto& block = builder.add_block(parent);
    auto& src = builder.add_access(block, "A");
    auto& dst = builder.add_access(block, "buf");

    tiles::TiledCopy plan;
    plan.atom = tiles::CopyAtom::VectorSync;
    plan.src = tiles::Layout({symbolic::integer(4)}, {symbolic::one()}, src_offset);
    plan.dst = tiles::Layout({symbolic::integer(4)}, {symbolic::one()}, symbolic::zero());
    auto& copy = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), data_flow::ImplementationType_NONE, plan, tiles::CopyDirection::In, 16
    ));
    builder.add_computational_memlet(block, dst, copy, "_dst", {}, ptr);
    builder.add_computational_memlet(block, src, copy, "_src", {}, ptr);
    return copy;
}

// Each of 32 kernel threads b loads 4 floats of A at @p src_offset into a register tile.
std::pair<tiles::TileCopyNode*, structured_control_flow::Map*>
add_vector_copy_in(builder::StructuredSDFGBuilder& builder, const symbolic::Expression& src_offset) {
    auto& map = add_gpu_map(builder, builder.subject().root(), "b", gpu::TargetLevel::X_BLOCK, 32);
    auto& copy = add_copy_in(builder, map.root(), src_offset);
    return {&copy, &map};
}

// Whether TileStreamingCopy on @p loop marks @p copy streaming.
bool marks_streaming(
    builder::StructuredSDFGBuilder& builder, structured_control_flow::StructuredLoop& loop, tiles::TileCopyNode& copy
) {
    analysis::AnalysisManager am(builder.subject());
    TileStreamingCopy transformation(loop);
    if (!transformation.can_be_applied(builder, am)) {
        return false;
    }
    transformation.apply(builder, am);
    return copy.streaming();
}

} // namespace

// Thread b reads A[4b, 4b+4): disjoint footprints, each element read once per launch.
TEST(TileStreamingCopyTest, MarksReadOnceCopyIn) {
    builder::StructuredSDFGBuilder builder("tsc_once", FunctionType_CPU);
    auto [copy, map] = add_vector_copy_in(builder, symbolic::mul(symbolic::integer(4), symbolic::symbol("b")));
    analysis::AnalysisManager am(builder.subject());
    EXPECT_FALSE(copy->streaming());

    TileStreamingCopy transformation(*map);
    ASSERT_TRUE(transformation.can_be_applied(builder, am));
    transformation.apply(builder, am);

    EXPECT_TRUE(copy->streaming());
    EXPECT_FALSE(transformation.can_be_applied(builder, am)) << "an already streaming copy is not re-marked";
}

// Every thread reads A[0, 4): the footprint is reused across the kernel, so it must stay cached.
TEST(TileStreamingCopyTest, RejectsReusedSource) {
    builder::StructuredSDFGBuilder builder("tsc_reused", FunctionType_CPU);
    auto [copy, map] = add_vector_copy_in(builder, symbolic::zero());
    analysis::AnalysisManager am(builder.subject());

    TileStreamingCopy transformation(*map);
    EXPECT_FALSE(transformation.can_be_applied(builder, am));
    EXPECT_FALSE(copy->streaming());
}

// Thread b reads A[2b, 2b+4): neighbouring threads overlap.
TEST(TileStreamingCopyTest, RejectsOverlappingThreadFootprints) {
    builder::StructuredSDFGBuilder builder("tsc_overlap", FunctionType_CPU);
    auto [copy, map] = add_vector_copy_in(builder, symbolic::mul(symbolic::integer(2), symbolic::symbol("b")));
    EXPECT_FALSE(marks_streaming(builder, *map, *copy));
}

// A sequential loop around the copy that does not move its footprint reads it again.
TEST(TileStreamingCopyTest, RejectsFootprintRepeatedByInnerLoop) {
    builder::StructuredSDFGBuilder builder("tsc_repeat", FunctionType_CPU);
    auto& map = add_gpu_map(builder, builder.subject().root(), "b", gpu::TargetLevel::X_BLOCK, 32);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        map.root(), i, symbolic::Lt(i, symbolic::integer(2)), symbolic::zero(), symbolic::add(i, symbolic::one())
    );
    auto& copy = add_copy_in(builder, loop.root(), symbolic::mul(symbolic::integer(4), symbolic::symbol("b")));
    EXPECT_FALSE(marks_streaming(builder, map, copy));
}

// Thread b, iteration i in [2b, 2b+2) reads A[4i, 4i+4): the inner loop starts at an outer
// indvar, and the folded footprints stay disjoint.
TEST(TileStreamingCopyTest, MarksInnerLoopStartingAtOuterIndvar) {
    builder::StructuredSDFGBuilder builder("tsc_inner", FunctionType_CPU);
    auto& map = add_gpu_map(builder, builder.subject().root(), "b", gpu::TargetLevel::X_BLOCK, 32);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto i = symbolic::symbol("i");
    auto start = symbolic::mul(symbolic::integer(2), symbolic::symbol("b"));
    auto& loop = builder.add_for(
        map.root(),
        i,
        symbolic::Lt(i, symbolic::add(start, symbolic::integer(2))),
        start,
        symbolic::add(i, symbolic::one())
    );
    auto& copy = add_copy_in(builder, loop.root(), symbolic::mul(symbolic::integer(4), i));
    EXPECT_TRUE(marks_streaming(builder, map, copy));
}

// An argument is fixed for the launch, so it may shift the origin.
TEST(TileStreamingCopyTest, MarksArgumentOrigin) {
    builder::StructuredSDFGBuilder builder("tsc_arg_origin", FunctionType_CPU);
    builder.add_container("n", types::Scalar(types::PrimitiveType::Int32), true);
    auto [copy, map] = add_vector_copy_in(
        builder,
        symbolic::add(symbolic::mul(symbolic::integer(4), symbolic::symbol("b")), symbolic::symbol("n"))
    );
    EXPECT_TRUE(marks_streaming(builder, *map, *copy));
}

// A non-argument origin may differ between threads.
TEST(TileStreamingCopyTest, RejectsNonUniformOrigin) {
    builder::StructuredSDFGBuilder builder("tsc_local_origin", FunctionType_CPU);
    builder.add_container("n", types::Scalar(types::PrimitiveType::Int32));
    auto [copy, map] = add_vector_copy_in(
        builder,
        symbolic::add(symbolic::mul(symbolic::integer(4), symbolic::symbol("b")), symbolic::symbol("n"))
    );
    EXPECT_FALSE(marks_streaming(builder, *map, *copy));
}

// An argument reassigned inside the kernel is no longer fixed for the launch.
TEST(TileStreamingCopyTest, RejectsArgumentOriginWrittenInKernel) {
    builder::StructuredSDFGBuilder builder("tsc_written_origin", FunctionType_CPU);
    builder.add_container("n", types::Scalar(types::PrimitiveType::Int32), true);
    auto n = symbolic::symbol("n");
    auto& map = add_gpu_map(builder, builder.subject().root(), "b", gpu::TargetLevel::X_BLOCK, 32);
    builder.add_assignments(map.root(), {{n, symbolic::symbol("b")}});
    auto& copy =
        add_copy_in(builder, map.root(), symbolic::add(symbolic::mul(symbolic::integer(4), symbolic::symbol("b")), n));
    EXPECT_FALSE(marks_streaming(builder, map, copy));
}

// The kernel launches 32 X_BLOCK threads for a sibling map; the copy outside it runs on all of
// them, so block g's footprint is read 32 times.
TEST(TileStreamingCopyTest, RejectsThreadsNotCoveredByEnclosingLoops) {
    builder::StructuredSDFGBuilder builder("tsc_uncovered", FunctionType_CPU);
    auto& grid = add_gpu_map(builder, builder.subject().root(), "g", gpu::TargetLevel::X_GRID, 4);
    auto& sibling = add_gpu_map(builder, grid.root(), "b", gpu::TargetLevel::X_BLOCK, 32);
    builder.add_block(sibling.root());
    auto& copy = add_copy_in(builder, grid.root(), symbolic::mul(symbolic::integer(4), symbolic::symbol("g")));
    EXPECT_FALSE(marks_streaming(builder, grid, copy));
}

// Lanes jointly execute one iteration, so every lane reads the same footprint.
TEST(TileStreamingCopyTest, RejectsLanes) {
    builder::StructuredSDFGBuilder builder("tsc_lanes", FunctionType_CPU);
    auto& map = add_gpu_map(builder, builder.subject().root(), "b", gpu::TargetLevel::X_BLOCK, 32, 32);
    auto& copy = add_copy_in(builder, map.root(), symbolic::mul(symbolic::integer(4), symbolic::symbol("b")));
    EXPECT_FALSE(marks_streaming(builder, map, copy));
}

// A While inside the kernel, around the transformed loop, repeats the copy.
TEST(TileStreamingCopyTest, RejectsWhileInKernel) {
    builder::StructuredSDFGBuilder builder("tsc_while", FunctionType_CPU);
    auto& map = add_gpu_map(builder, builder.subject().root(), "b", gpu::TargetLevel::X_BLOCK, 32);
    auto& repeat = builder.add_while(map.root());
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        repeat.root(), i, symbolic::Lt(i, symbolic::one()), symbolic::zero(), symbolic::add(i, symbolic::one())
    );
    auto& copy = add_copy_in(builder, loop.root(), symbolic::mul(symbolic::integer(4), symbolic::symbol("b")));
    EXPECT_FALSE(marks_streaming(builder, loop, copy));
}
