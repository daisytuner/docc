// Unit tests for the (non-deprecated) CUDAOffloadTransform::can_be_applied.
//
// These cover the three grid target levels (X/Y/Z), the parallel-size hardware
// limits enforced per level, and static vs. dynamic argument sizes. Both the
// accepting (positive) and rejecting (negative) cases are exercised.

#include "sdfg/transformations/offloading/cuda_offload_transform.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg::cuda {

namespace {

// X grid dimension is limited to 2^31 - 1, Y and Z grid dimensions to 2^16 - 1.
constexpr int64_t kMaxGridX = 2147483647; // 2^31 - 1
constexpr int64_t kMaxGridYZ = 65535; // 2^16 - 1

// Builds a minimal, valid offloadable map that writes `A[i]` for i in [0, bound).
// `bound` may be a constant (static size) or a symbol (dynamic size); when it is
// a symbol, the caller must add the corresponding container first.
structured_control_flow::Map& build_offloadable_map(builder::StructuredSDFGBuilder& builder, symbolic::Expression bound) {
    auto& root = builder.subject().root();

    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer f32ptr(f32);
    types::Scalar i64(types::PrimitiveType::Int64);

    builder.add_container("i", i64);
    builder.add_container("A", f32ptr, /*is_argument=*/true);

    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), bound),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(map.root());
    auto& write = builder.add_access(block, "A");
    auto& constant = builder.add_constant(block, "0.0f", f32);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, f32);
    builder.add_computational_memlet(block, tasklet, "out_", write, {symbolic::symbol("i")});

    return map;
}

// Builds a map whose written argument is indexed by a free local `k` (only
// type-default bounds), so the access cannot be bounded and the argument size
// is genuinely unknown (as opposed to a dynamic-but-known symbolic size).
structured_control_flow::Map& build_unknown_size_map(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();

    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer f32ptr(f32);
    types::Scalar i64(types::PrimitiveType::Int64);

    builder.add_container("i", i64);
    builder.add_container("k", i64);
    builder.add_container("A", f32ptr, /*is_argument=*/true);

    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(1024)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );

    auto& block = builder.add_block(map.root());
    auto& write = builder.add_access(block, "A");
    auto& constant = builder.add_constant(block, "0.0f", f32);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, f32);
    builder.add_computational_memlet(block, tasklet, "out_", write, {symbolic::symbol("k")});

    return map;
}

// Counts CUDA data-offloading library nodes across the top-level blocks that
// apply produces (host-side alloc/copy/free around the offloaded map).
size_t count_offloading_nodes(builder::StructuredSDFGBuilder& builder) {
    auto& root = builder.subject().root();
    size_t count = 0;
    for (size_t i = 0; i < root.size(); ++i) {
        auto& cf_node = root.at(i);
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&cf_node)) {
            for (auto& node : block->dataflow().nodes()) {
                if (dynamic_cast<CUDADataOffloadingNode*>(&node) != nullptr) {
                    ++count;
                }
            }
        }
    }
    return count;
}

// Applies the transform and asserts the resulting map schedule is a CUDA
// offloader schedule that preserves the intended target level and parallel size.
void expect_schedule_retained(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    CUDAOffloadTransform& transform,
    structured_control_flow::Map& map,
    gpu::TargetLevel expected_level,
    int64_t expected_parallel_size
) {
    ASSERT_TRUE(transform.can_be_applied(builder, analysis_manager));
    transform.apply(builder, analysis_manager);

    const auto& schedule = map.schedule_type();
    EXPECT_EQ(schedule.value(), ScheduleType_CUDA_Offload::value());
    EXPECT_EQ(ScheduleType_CUDA_Offload::target_level(schedule), expected_level);
    EXPECT_EQ(ScheduleType_CUDA_Offload::parallel_size(schedule)->as_int(), expected_parallel_size);

    // The offloaded argument must be surrounded by CUDA data-offloading nodes.
    EXPECT_GT(count_offloading_nodes(builder), 0u);
}

} // namespace

// --- Grid target levels: positive (within limit) ---------------------------

TEST(CUDAOffloadTransformTest, XGridWithinLimit) {
    builder::StructuredSDFGBuilder builder("cuda_xgrid_ok", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(kMaxGridX), gpu::TargetLevel::X_GRID);

    expect_schedule_retained(builder, analysis_manager, transform, map, gpu::TargetLevel::X_GRID, kMaxGridX);
}

TEST(CUDAOffloadTransformTest, YGridWithinLimit) {
    builder::StructuredSDFGBuilder builder("cuda_ygrid_ok", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(kMaxGridYZ), gpu::TargetLevel::Y_GRID);

    expect_schedule_retained(builder, analysis_manager, transform, map, gpu::TargetLevel::Y_GRID, kMaxGridYZ);
}

TEST(CUDAOffloadTransformTest, ZGridWithinLimit) {
    builder::StructuredSDFGBuilder builder("cuda_zgrid_ok", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(kMaxGridYZ), gpu::TargetLevel::Z_GRID);

    expect_schedule_retained(builder, analysis_manager, transform, map, gpu::TargetLevel::Z_GRID, kMaxGridYZ);
}

// --- Grid target levels: negative (exceeds limit) --------------------------

TEST(CUDAOffloadTransformTest, XGridExceedsLimit) {
    builder::StructuredSDFGBuilder builder("cuda_xgrid_over", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(kMaxGridX + 1), gpu::TargetLevel::X_GRID);

    EXPECT_FALSE(transform.can_be_applied(builder, analysis_manager));
}

TEST(CUDAOffloadTransformTest, YGridExceedsLimit) {
    builder::StructuredSDFGBuilder builder("cuda_ygrid_over", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(kMaxGridYZ + 1), gpu::TargetLevel::Y_GRID);

    EXPECT_FALSE(transform.can_be_applied(builder, analysis_manager));
}

TEST(CUDAOffloadTransformTest, ZGridExceedsLimit) {
    builder::StructuredSDFGBuilder builder("cuda_zgrid_over", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(kMaxGridYZ + 1), gpu::TargetLevel::Z_GRID);

    EXPECT_FALSE(transform.can_be_applied(builder, analysis_manager));
}

// --- Unsupported (non-grid) target levels: negative ------------------------

TEST(CUDAOffloadTransformTest, BlockTargetLevelRejected) {
    builder::StructuredSDFGBuilder builder("cuda_block_level", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(32), gpu::TargetLevel::X_BLOCK);

    EXPECT_FALSE(transform.can_be_applied(builder, analysis_manager));
}

TEST(CUDAOffloadTransformTest, WarpTargetLevelRejected) {
    builder::StructuredSDFGBuilder builder("cuda_warp_level", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(32), gpu::TargetLevel::WARP);

    EXPECT_FALSE(transform.can_be_applied(builder, analysis_manager));
}

// --- Parallel sizes --------------------------------------------------------

TEST(CUDAOffloadTransformTest, SmallParallelSizeAccepted) {
    builder::StructuredSDFGBuilder builder("cuda_small_parallel", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(32), gpu::TargetLevel::X_GRID);

    expect_schedule_retained(builder, analysis_manager, transform, map, gpu::TargetLevel::X_GRID, 32);
}

TEST(CUDAOffloadTransformTest, YGridParallelSizeBoundaryVsOverflow) {
    // Exactly at the limit is accepted, one over is rejected.
    {
        builder::StructuredSDFGBuilder builder("cuda_y_boundary", FunctionType_CPU);
        auto& map = build_offloadable_map(builder, symbolic::integer(1024));
        analysis::AnalysisManager analysis_manager(builder.subject());
        CUDAOffloadTransform transform(map, symbolic::integer(kMaxGridYZ), gpu::TargetLevel::Y_GRID);
        expect_schedule_retained(builder, analysis_manager, transform, map, gpu::TargetLevel::Y_GRID, kMaxGridYZ);
    }
    {
        builder::StructuredSDFGBuilder builder("cuda_y_overflow", FunctionType_CPU);
        auto& map = build_offloadable_map(builder, symbolic::integer(1024));
        analysis::AnalysisManager analysis_manager(builder.subject());
        CUDAOffloadTransform transform(map, symbolic::integer(kMaxGridYZ + 1), gpu::TargetLevel::Y_GRID);
        EXPECT_FALSE(transform.can_be_applied(builder, analysis_manager));
    }
}

// --- Static vs. dynamic argument sizes -------------------------------------

TEST(CUDAOffloadTransformTest, StaticSizeAccepted) {
    builder::StructuredSDFGBuilder builder("cuda_static_size", FunctionType_CPU);
    auto& map = build_offloadable_map(builder, symbolic::integer(1024));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform
        transform(map, symbolic::integer(1024), gpu::TargetLevel::X_GRID, /*allow_dynamic_sizes=*/false);

    expect_schedule_retained(builder, analysis_manager, transform, map, gpu::TargetLevel::X_GRID, 1024);
}

TEST(CUDAOffloadTransformTest, DynamicSizeAcceptedWhenAllowed) {
    builder::StructuredSDFGBuilder builder("cuda_dynamic_size", FunctionType_CPU);
    types::Scalar i64(types::PrimitiveType::Int64);
    builder.add_container("N", i64, /*is_argument=*/true);
    auto& map = build_offloadable_map(builder, symbolic::symbol("N"));

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(1024), gpu::TargetLevel::X_GRID, /*allow_dynamic_sizes=*/true);

    expect_schedule_retained(builder, analysis_manager, transform, map, gpu::TargetLevel::X_GRID, 1024);
}

TEST(CUDAOffloadTransformTest, UnknownSizeRejected) {
    builder::StructuredSDFGBuilder builder("cuda_unknown_size", FunctionType_CPU);
    auto& map = build_unknown_size_map(builder);

    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAOffloadTransform transform(map, symbolic::integer(1024), gpu::TargetLevel::X_GRID, /*allow_dynamic_sizes=*/true);

    EXPECT_FALSE(transform.can_be_applied(builder, analysis_manager));
}

} // namespace sdfg::cuda
