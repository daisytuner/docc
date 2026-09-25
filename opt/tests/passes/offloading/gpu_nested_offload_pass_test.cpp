#include "sdfg/passes/offloading/gpu_nested_offload_pass.h"

#include <gtest/gtest.h>

#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/rocm/rocm.h"

using namespace sdfg;

namespace {

const std::string kBuffer = "__daisy_A";

symbolic::Condition lt(const std::string& v, int64_t bound) {
    return symbolic::Lt(symbolic::symbol(v), symbolic::integer(bound));
}

symbolic::Expression step(const std::string& v) {
    return symbolic::add(symbolic::symbol(v), symbolic::integer(1));
}

// X_GRID schedule for the pre-offloaded outer map of the given GPU target.
template<typename Sched>
structured_control_flow::ScheduleType grid_schedule(int64_t size) {
    return Sched::template create<Sched>(gpu::TargetLevel::X_GRID, symbolic::integer(size));
}

// Flat index ((v0*b1 + v1)*b2 + v2)... that is unique per thread, so folding any
// nested loop into a parallel dimension stays race-free.
symbolic::Expression flat_index(const std::vector<std::string>& vars, const std::vector<int64_t>& bounds) {
    symbolic::Expression index = symbolic::symbol(vars[0]);
    for (size_t i = 1; i < vars.size(); ++i) {
        index = symbolic::add(symbolic::mul(index, symbolic::integer(bounds[i])), symbolic::symbol(vars[i]));
    }
    return index;
}

// Stores 0.0f into kBuffer[index] within `parent`.
void add_store(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const symbolic::Expression& index
) {
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    auto& block = builder.add_block(parent);
    auto& access = builder.add_access(block, kBuffer);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& constant = builder.add_constant(block, "0.0f", base_desc);
    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, base_desc);
    builder.add_computational_memlet(block, tasklet, "out_", access, {index}, pointer_type);
}

void add_index_containers(builder::StructuredSDFGBuilder& builder, const std::vector<std::string>& vars) {
    types::Scalar int_desc(types::PrimitiveType::Int32);
    for (const auto& v : vars) {
        builder.add_container(v, int_desc);
    }
    types::Pointer pointer_type((types::Scalar(types::PrimitiveType::Float)));
    builder.add_container(kBuffer, pointer_type);
}

bool is_sequential(const structured_control_flow::StructuredLoop& loop) {
    return loop.schedule_type().value() == structured_control_flow::ScheduleType_Sequential::value();
}

} // namespace

TEST(GPUNestedOffloadPassTest, EmptyLoops_ReturnsFalse) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    std::vector<structured_control_flow::StructuredLoop*> loops;
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::CUDA);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(GPUNestedOffloadPassTest, SingleLoopDepthTooShallow_ReturnsFalse) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    auto& root = builder.subject().root();
    add_index_containers(builder, {"i"});

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        grid_schedule<cuda::ScheduleType_CUDA_Offload>(128)
    );
    add_store(builder, grid.root(), symbolic::symbol("i"));

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&grid};
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::CUDA);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

// ---- Consideration 2: explicit CUDA vs ROCm coverage -----------------------

TEST(GPUNestedOffloadPassTest, Depth2_XBlock_CUDA) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    auto& root = builder.subject().root();
    add_index_containers(builder, {"i", "j"});

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        grid_schedule<cuda::ScheduleType_CUDA_Offload>(128)
    );
    auto& inner = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 64),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_store(builder, inner.root(), flat_index({"i", "j"}, {128, 64}));

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&grid};
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::CUDA);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(inner.schedule_type().value(), cuda::ScheduleType_CUDA_Offload::value());
    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(inner.schedule_type()), gpu::TargetLevel::X_BLOCK);
    EXPECT_TRUE(symbolic::eq(cuda::ScheduleType_CUDA_Offload::parallel_size(inner.schedule_type()), symbolic::integer(64))
    );
}

TEST(GPUNestedOffloadPassTest, Depth2_XBlock_ROCM) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    auto& root = builder.subject().root();
    add_index_containers(builder, {"i", "j"});

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        grid_schedule<rocm::ScheduleType_ROCM_Offload>(128)
    );
    auto& inner = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 64),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_store(builder, inner.root(), flat_index({"i", "j"}, {128, 64}));

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&grid};
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::ROCM);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(inner.schedule_type().value(), rocm::ScheduleType_ROCM_Offload::value());
    EXPECT_EQ(rocm::ScheduleType_ROCM_Offload::target_level(inner.schedule_type()), gpu::TargetLevel::X_BLOCK);
}

// ---- Consideration 1: depth-3 WARP budget behavior (current behavior) -------

// chain[0] has 32 iterations => X_BLOCK(32); X_BLOCK * WARP = 32 * 32 = 1024 <= 1024
// so the WARP level also applies.
TEST(GPUNestedOffloadPassTest, Depth3_ChainFitsWarpBudget_XBlockAndWarpApplied) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    auto& root = builder.subject().root();
    add_index_containers(builder, {"i", "j", "k"});

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        grid_schedule<cuda::ScheduleType_CUDA_Offload>(128)
    );
    auto& c0 = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 32),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& c1 = builder.add_map(
        c0.root(),
        symbolic::symbol("k"),
        lt("k", 32),
        symbolic::zero(),
        step("k"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_store(builder, c1.root(), flat_index({"i", "j", "k"}, {128, 32, 32}));

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&grid};
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::CUDA);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(c0.schedule_type()), gpu::TargetLevel::X_BLOCK);
    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(c1.schedule_type()), gpu::TargetLevel::WARP);
}

// chain[0] has 128 iterations => X_BLOCK(128); X_BLOCK * WARP = 128 * 32 = 4096 > 1024
// so the WARP level is rejected and only X_BLOCK applies.
TEST(GPUNestedOffloadPassTest, Depth3_XBlockExceedsWarpBudget_OnlyXBlockApplied) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    auto& root = builder.subject().root();
    add_index_containers(builder, {"i", "j", "k"});

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        grid_schedule<cuda::ScheduleType_CUDA_Offload>(128)
    );
    auto& c0 = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 128),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& c1 = builder.add_map(
        c0.root(),
        symbolic::symbol("k"),
        lt("k", 32),
        symbolic::zero(),
        step("k"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_store(builder, c1.root(), flat_index({"i", "j", "k"}, {128, 128, 32}));

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&grid};
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::CUDA);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(c0.schedule_type()), gpu::TargetLevel::X_BLOCK);
    EXPECT_TRUE(is_sequential(c1));
}

// The pass hardcodes WARP size 32, but ROCm's wavefront is 64, so the depth-3 WARP
// level is always rejected on ROCm and only X_BLOCK applies.
TEST(GPUNestedOffloadPassTest, Depth3_ROCM_WarpRejected_OnlyXBlock) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    auto& root = builder.subject().root();
    add_index_containers(builder, {"i", "j", "k"});

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        grid_schedule<rocm::ScheduleType_ROCM_Offload>(128)
    );
    auto& c0 = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 32),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& c1 = builder.add_map(
        c0.root(),
        symbolic::symbol("k"),
        lt("k", 32),
        symbolic::zero(),
        step("k"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_store(builder, c1.root(), flat_index({"i", "j", "k"}, {128, 32, 32}));

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&grid};
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::ROCM);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(rocm::ScheduleType_ROCM_Offload::target_level(c0.schedule_type()), gpu::TargetLevel::X_BLOCK);
    EXPECT_TRUE(is_sequential(c1));
}

// ---- Deeper nest and rejection paths ---------------------------------------

TEST(GPUNestedOffloadPassTest, Depth4_YGridXBlockYBlock) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    auto& root = builder.subject().root();
    add_index_containers(builder, {"i", "j", "k", "l"});

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        grid_schedule<cuda::ScheduleType_CUDA_Offload>(128)
    );
    auto& c0 = builder.add_map(
        grid.root(),
        symbolic::symbol("j"),
        lt("j", 100),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& c1 = builder.add_map(
        c0.root(),
        symbolic::symbol("k"),
        lt("k", 8),
        symbolic::zero(),
        step("k"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& c2 = builder.add_map(
        c1.root(),
        symbolic::symbol("l"),
        lt("l", 4),
        symbolic::zero(),
        step("l"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_store(builder, c2.root(), flat_index({"i", "j", "k", "l"}, {128, 100, 8, 4}));

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&grid};
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::CUDA);
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(c0.schedule_type()), gpu::TargetLevel::Y_GRID);
    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(c1.schedule_type()), gpu::TargetLevel::X_BLOCK);
    EXPECT_EQ(cuda::ScheduleType_CUDA_Offload::target_level(c2.schedule_type()), gpu::TargetLevel::Y_BLOCK);

    // Y_GRID uses the exact iteration count.
    EXPECT_TRUE(symbolic::eq(cuda::ScheduleType_CUDA_Offload::parallel_size(c0.schedule_type()), symbolic::integer(100))
    );
}

TEST(GPUNestedOffloadPassTest, OuterNotOffloaded_ReturnsFalse) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    auto& root = builder.subject().root();
    add_index_containers(builder, {"i", "j"});

    auto& outer = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& inner = builder.add_map(
        outer.root(),
        symbolic::symbol("j"),
        lt("j", 64),
        symbolic::zero(),
        step("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    add_store(builder, inner.root(), flat_index({"i", "j"}, {128, 64}));

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&outer};
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::CUDA);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
    EXPECT_TRUE(is_sequential(inner));
}

TEST(GPUNestedOffloadPassTest, NestedPlainFor_ReturnsFalse) {
    builder::StructuredSDFGBuilder builder("t", FunctionType_CPU);
    auto& root = builder.subject().root();
    add_index_containers(builder, {"i", "j"});

    auto& grid = builder.add_map(
        root,
        symbolic::symbol("i"),
        lt("i", 128),
        symbolic::zero(),
        step("i"),
        grid_schedule<cuda::ScheduleType_CUDA_Offload>(128)
    );
    builder.add_for(grid.root(), symbolic::symbol("j"), lt("j", 64), symbolic::zero(), step("j"));

    analysis::AnalysisManager analysis_manager(builder.subject());
    std::vector<structured_control_flow::StructuredLoop*> loops{&grid};
    passes::GPUNestedOffloadPass pass(loops, passes::GPUTarget::CUDA);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}
