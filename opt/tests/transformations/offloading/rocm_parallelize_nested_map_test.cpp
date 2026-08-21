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

// ---------------------------------------------------------------------------
// Sibling accumulation replication (softmax-style kernel): an outer X-dimension
// ROCm map whose body contains a sibling *reduce* loop that accumulates into a
// shared buffer (`acc[i] = acc[i] + P[...]`, a read-modify-write) next to a plain
// *map* loop. Folding a new grid dimension for the map loop replicates the reduce
// across the new dimension's threads, which race on the shared accumulator.
// Parallelizing the map loop must therefore be rejected - unless the accumulator
// is privatizable (loop-local) or the reduction is itself parallelized.
// ---------------------------------------------------------------------------

TEST(ROCMNestedParallelismTransformation, SiblingAccumulationBlocksNestedParallelism) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("r", int_desc);
    builder.add_container("m", int_desc);
    builder.add_container("P", pointer_type, true); // input argument
    builder.add_container("acc", pointer_type, true); // reduction accumulator (escapes the kernel)
    builder.add_container("Out", pointer_type, true); // output argument

    auto i = symbolic::symbol("i");
    auto r = symbolic::symbol("r");
    auto m = symbolic::symbol("m");

    ScheduleType outer_schedule = ScheduleType_ROCM::create();
    ScheduleType_ROCM::dimension(outer_schedule, ROCMDimension::X);
    ScheduleType_ROCM::block_size(outer_schedule, symbolic::integer(64));
    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(16384)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        outer_schedule
    );

    // Sibling 1: sequential reduce loop accumulating into acc[i] (read-modify-write).
    auto& reduce = builder.add_map(
        outer.root(),
        r,
        symbolic::Lt(r, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(r, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    {
        auto& block = builder.add_block(reduce.root());
        auto& acc_in = builder.add_access(block, "acc");
        auto& p_in = builder.add_access(block, "P");
        auto& acc_out = builder.add_access(block, "acc");
        auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, acc_in, tk, "_in1", {i}, pointer_type);
        builder.add_computational_memlet(
            block, p_in, tk, "_in2", {symbolic::add(symbolic::mul(i, symbolic::integer(32)), r)}, pointer_type
        );
        builder.add_computational_memlet(block, tk, "_out", acc_out, {i}, pointer_type);
    }

    // Sibling 2 (target): sequential map loop doing a plain store.
    auto& target = builder.add_map(
        outer.root(),
        m,
        symbolic::Lt(m, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(m, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    {
        auto& block = builder.add_block(target.root());
        auto& p_in = builder.add_access(block, "P");
        auto& out = builder.add_access(block, "Out");
        auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
        auto idx = symbolic::add(symbolic::mul(i, symbolic::integer(32)), m);
        builder.add_computational_memlet(block, p_in, tk, "in_", {idx}, pointer_type);
        builder.add_computational_memlet(block, tk, "out_", out, {idx}, pointer_type);
    }

    transformations::ROCMParallelizeNestedMap transformation(target, 4);
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(ROCMNestedParallelismTransformation, SiblingAccumulationOnLocalAllowsNestedParallelism) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("r", int_desc);
    builder.add_container("m", int_desc);
    builder.add_container("P", pointer_type, true); // input argument
    builder.add_container("acc", pointer_type); // transient used only inside the kernel -> privatizable local
    builder.add_container("Out", pointer_type, true); // output argument

    auto i = symbolic::symbol("i");
    auto r = symbolic::symbol("r");
    auto m = symbolic::symbol("m");

    ScheduleType outer_schedule = ScheduleType_ROCM::create();
    ScheduleType_ROCM::dimension(outer_schedule, ROCMDimension::X);
    ScheduleType_ROCM::block_size(outer_schedule, symbolic::integer(64));
    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(16384)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        outer_schedule
    );

    auto& reduce = builder.add_map(
        outer.root(),
        r,
        symbolic::Lt(r, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(r, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    {
        auto& block = builder.add_block(reduce.root());
        auto& acc_in = builder.add_access(block, "acc");
        auto& p_in = builder.add_access(block, "P");
        auto& acc_out = builder.add_access(block, "acc");
        auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, acc_in, tk, "_in1", {i}, pointer_type);
        builder.add_computational_memlet(
            block, p_in, tk, "_in2", {symbolic::add(symbolic::mul(i, symbolic::integer(32)), r)}, pointer_type
        );
        builder.add_computational_memlet(block, tk, "_out", acc_out, {i}, pointer_type);
    }

    auto& target = builder.add_map(
        outer.root(),
        m,
        symbolic::Lt(m, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(m, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    {
        auto& block = builder.add_block(target.root());
        auto& p_in = builder.add_access(block, "P");
        auto& out = builder.add_access(block, "Out");
        auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
        auto idx = symbolic::add(symbolic::mul(i, symbolic::integer(32)), m);
        builder.add_computational_memlet(block, p_in, tk, "in_", {idx}, pointer_type);
        builder.add_computational_memlet(block, tk, "out_", out, {idx}, pointer_type);
    }

    transformations::ROCMParallelizeNestedMap transformation(target, 4);
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(ROCMNestedParallelismTransformation, ParallelizedSiblingReductionAllowsNestedParallelism) {
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("r", int_desc);
    builder.add_container("m", int_desc);
    builder.add_container("P", pointer_type, true); // input argument
    builder.add_container("acc", pointer_type, true); // escapes, but the reduce is parallelized
    builder.add_container("Out", pointer_type, true); // output argument

    auto i = symbolic::symbol("i");
    auto r = symbolic::symbol("r");
    auto m = symbolic::symbol("m");

    ScheduleType outer_schedule = ScheduleType_ROCM::create();
    ScheduleType_ROCM::dimension(outer_schedule, ROCMDimension::X);
    ScheduleType_ROCM::block_size(outer_schedule, symbolic::integer(64));
    auto& outer = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::integer(16384)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        outer_schedule
    );

    // Sibling 1: reduce that is itself parallelized (a ROCm map) - mapped onto its
    // own threads, so it is not replicated by the target's new dimension.
    ScheduleType reduce_schedule = ScheduleType_ROCM::create();
    ScheduleType_ROCM::dimension(reduce_schedule, ROCMDimension::Y);
    ScheduleType_ROCM::block_size(reduce_schedule, symbolic::integer(8));
    auto& reduce = builder.add_map(
        outer.root(),
        r,
        symbolic::Lt(r, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(r, symbolic::integer(1)),
        reduce_schedule
    );
    {
        auto& block = builder.add_block(reduce.root());
        auto& acc_in = builder.add_access(block, "acc");
        auto& p_in = builder.add_access(block, "P");
        auto& acc_out = builder.add_access(block, "acc");
        auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, acc_in, tk, "_in1", {i}, pointer_type);
        builder.add_computational_memlet(
            block, p_in, tk, "_in2", {symbolic::add(symbolic::mul(i, symbolic::integer(32)), r)}, pointer_type
        );
        builder.add_computational_memlet(block, tk, "_out", acc_out, {i}, pointer_type);
    }

    auto& target = builder.add_map(
        outer.root(),
        m,
        symbolic::Lt(m, symbolic::integer(32)),
        symbolic::integer(0),
        symbolic::add(m, symbolic::integer(1)),
        ScheduleType_Sequential::create()
    );
    {
        auto& block = builder.add_block(target.root());
        auto& p_in = builder.add_access(block, "P");
        auto& out = builder.add_access(block, "Out");
        auto& tk = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
        auto idx = symbolic::add(symbolic::mul(i, symbolic::integer(32)), m);
        builder.add_computational_memlet(block, p_in, tk, "in_", {idx}, pointer_type);
        builder.add_computational_memlet(block, tk, "out_", out, {idx}, pointer_type);
    }

    transformations::ROCMParallelizeNestedMap transformation(target, 4);
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_TRUE(transformation.can_be_applied(builder, analysis_manager));
}

} // namespace sdfg::rocm
