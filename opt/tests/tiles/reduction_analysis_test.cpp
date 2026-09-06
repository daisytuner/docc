#include "sdfg/tiles/analysis/reduction_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

// is_reduction_accumulator detects it whether the Reduce is the loop itself, an
// ancestor, or a descendant.
TEST(ReductionAnalysisTest, IsReductionAccumulator_EnclosingAndNested) {
    builder::StructuredSDFGBuilder builder("ls_reduce_acc", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("i", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);
    builder.add_container("other", ptr);

    auto& for_j =
        builder.add_for(seq, j, symbolic::Lt(j, N), symbolic::integer(0), symbolic::add(j, symbolic::integer(1)));
    auto& reduce_i = builder.add_reduce(
        for_j.root(),
        i,
        symbolic::Lt(i, N),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& loop_k =
        builder
            .add_for(reduce_i.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    EXPECT_TRUE(tiles::is_reduction_accumulator(reduce_i, "acc", am)); // the reduce itself
    EXPECT_TRUE(tiles::is_reduction_accumulator(loop_k, "acc", am)); // ancestor reduce
    EXPECT_TRUE(tiles::is_reduction_accumulator(for_j, "acc", am)); // descendant reduce
    EXPECT_FALSE(tiles::is_reduction_accumulator(loop_k, "other", am));
}

// collect_reduction_owners: a sequential (non-cooperative) Reduce at the localized
// loop is privatizable — it is returned so apply() can retarget its descriptor.
TEST(ReductionAnalysisTest, CollectReductionOwners_SequentialAccepts) {
    builder::StructuredSDFGBuilder builder("ls_collect_seq", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("acc", ptr);

    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_TRUE(tiles::collect_reduction_owners(reduce_j, "acc", am, owners));
    ASSERT_EQ(owners.size(), 1u);
    EXPECT_EQ(owners.front(), &reduce_j);
}

// collect_reduction_owners: a GPU-offloaded (cooperatively combined) Reduce is
// owned by the reduce dispatcher — reject.
TEST(ReductionAnalysisTest, CollectReductionOwners_CooperativeRejects) {
    builder::StructuredSDFGBuilder builder("ls_collect_coop", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    builder.add_container("N", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("acc", ptr);

    auto block = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        block
    );

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_FALSE(tiles::collect_reduction_owners(reduce_j, "acc", am, owners));
}

// collect_reduction_owners: a *sequential* ancestor Reduce is accepted — its outer
// iterations are barrier-separated, so a read-modify-write copy-in/out around the
// localized loop carries the accumulation through the global container each
// iteration (classical BLIS pc loop). The ancestor is not retargeted (owners empty).
TEST(ReductionAnalysisTest, CollectReductionOwners_SequentialAncestorAccepts) {
    builder::StructuredSDFGBuilder builder("ls_collect_ancestor_seq", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);

    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& loop_k =
        builder
            .add_for(reduce_j.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_TRUE(tiles::collect_reduction_owners(loop_k, "acc", am, owners));
    EXPECT_TRUE(owners.empty());
}

// collect_reduction_owners: a GPU block-cooperative ancestor Reduce is combined by
// the reduce dispatcher (not an atomic-merge grid reduce), so localizing its
// accumulator at an inner loop is rejected.
TEST(ReductionAnalysisTest, CollectReductionOwners_GpuBlockAncestorRejects) {
    builder::StructuredSDFGBuilder builder("ls_collect_ancestor_block", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);

    auto block = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        block
    );
    auto& loop_k =
        builder
            .add_for(reduce_j.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_FALSE(tiles::collect_reduction_owners(loop_k, "acc", am, owners));
}

// collect_reduction_owners: a grid-parallel (split-K Z_GRID) ancestor Reduce merges
// per-block partials via an atomic writeback, so privatizing the per-block partial
// into a register tile at an inner loop is permitted — and the reduce is NOT
// retargeted (the cross-block merge is taken over separately).
TEST(ReductionAnalysisTest, CollectReductionOwners_GridParallelAncestorAccepts) {
    builder::StructuredSDFGBuilder builder("ls_collect_grid_ancestor", FunctionType_CPU);
    auto& seq = builder.subject().root();
    types::Scalar loop_var(types::PrimitiveType::Int32);
    types::Pointer ptr(types::StorageType::NV_Generic(), 0, "", types::Scalar(types::PrimitiveType::Float));
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto K = symbolic::symbol("K");
    builder.add_container("N", loop_var, true);
    builder.add_container("K", loop_var, true);
    builder.add_container("j", loop_var);
    builder.add_container("k", loop_var);
    builder.add_container("acc", ptr);

    auto grid = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::Z_GRID, symbolic::integer(16));
    auto& reduce_j = builder.add_reduce(
        seq,
        j,
        symbolic::Lt(j, N),
        symbolic::integer(0),
        symbolic::add(j, symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "acc"}},
        grid
    );
    auto& loop_k =
        builder
            .add_for(reduce_j.root(), k, symbolic::Lt(k, K), symbolic::integer(0), symbolic::add(k, symbolic::integer(1)));

    analysis::AnalysisManager am(builder.subject());
    std::vector<structured_control_flow::Reduce*> owners;
    EXPECT_TRUE(tiles::collect_reduction_owners(loop_k, "acc", am, owners));
    EXPECT_TRUE(owners.empty());
}
