#include "sdfg/passes/scheduler/omp_scheduler.h"

#include "sdfg/scheduling/loops/loop_scheduling_pass.h"
#include "sdfg/targets/omp/schedule.h"
#include "sdfg/transformations/recorder.h"

#include <gtest/gtest.h>

using namespace sdfg;

static passes::scheduler::OMPScheduler* get_omp_sched() {
    static passes::scheduler::OMPScheduler instance;
    return &instance;
}

TEST(OMPSchedulerTest, OuterParallelMapWithInnerMap) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Define loop 1
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");

    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = loop.root();

    // Define loop 2
    auto bound_2 = symbolic::symbol("M");
    auto indvar_2 = symbolic::symbol("j");

    auto& loop_2 = builder.add_map(
        body,
        indvar_2,
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("M")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_2 = loop_2.root();

    // Add computation
    auto& block = builder.add_block(body_2);
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder
        .add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);
    builder
        .add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i"), symbolic::symbol("j")}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());

    scheduling::loops::LoopSchedulingPass loop_scheduling_pass({get_omp_sched()}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    auto& schedule_loop = sdfg.root().at(0);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Map*>(&schedule_loop));
    auto& scheduled_map = static_cast<structured_control_flow::Map&>(schedule_loop);

    EXPECT_EQ(scheduled_map.schedule_type().value(), omp::ScheduleType_OMP::value());
}

TEST(OMPSchedulerTest, OuterWhileWith2DMap) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array desc_1(base_desc, symbolic::symbol("M"));
    types::Pointer desc_2(desc_1);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    // Define loop 1
    auto& loop = builder.add_while(root);
    auto& body = loop.root();


    // Define loop 2
    auto bound_2 = symbolic::symbol("M");
    auto indvar_2 = symbolic::symbol("j");

    auto& loop_2 = builder.add_map(
        body,
        indvar_2,
        symbolic::Lt(indvar_2, bound_2),
        symbolic::integer(0),
        symbolic::add(indvar_2, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_2 = loop_2.root();

    // Define loop 3
    auto bound_3 = symbolic::symbol("K");
    auto indvar_3 = symbolic::symbol("k");

    auto& loop_3 = builder.add_map(
        body_2,
        indvar_3,
        symbolic::Lt(indvar_3, bound_3),
        symbolic::integer(0),
        symbolic::add(indvar_3, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_3 = loop_3.root();

    // Add computation
    {
        auto& block = builder.add_block(body_3);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar_2, indvar_3}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar_2, indvar_3}, desc_2);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());

    scheduling::loops::LoopSchedulingPass loop_scheduling_pass({get_omp_sched()}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    auto& schedule_loop = loop.root().at(0);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Map*>(&schedule_loop));

    auto& scheduled_map = static_cast<structured_control_flow::Map&>(schedule_loop);
    EXPECT_EQ(scheduled_map.schedule_type().value(), omp::ScheduleType_OMP::value());
}

TEST(OMPSchedulerTest, OuterWhileWithInnerMaps) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc_2(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("M", sym_desc, true);
    builder.add_container("K", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);

    // Define loop 1
    auto& loop = builder.add_while(root);
    auto& body = loop.root();

    // Define loop 2
    auto bound_2 = symbolic::symbol("M");
    auto indvar_2 = symbolic::symbol("j");

    auto& loop_2 = builder.add_map(
        body,
        indvar_2,
        symbolic::Lt(indvar_2, bound_2),
        symbolic::integer(0),
        symbolic::add(indvar_2, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_2 = loop_2.root();

    // Add computation
    {
        auto& block = builder.add_block(body_2);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar_2}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar_2}, desc_2);
    }

    // Define loop 3
    auto bound_3 = symbolic::symbol("K");
    auto indvar_3 = symbolic::symbol("k");

    auto& loop_3 = builder.add_map(
        body,
        indvar_3,
        symbolic::Lt(indvar_3, bound_3),
        symbolic::integer(0),
        symbolic::add(indvar_3, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body_3 = loop_3.root();

    // Add computation
    {
        auto& block = builder.add_block(body_3);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar_3}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar_3}, desc_2);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());

    scheduling::loops::LoopSchedulingPass loop_scheduling_pass({get_omp_sched()}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    EXPECT_EQ(loop_2.schedule_type().value(), omp::ScheduleType_OMP::value());
    EXPECT_EQ(loop_3.schedule_type().value(), omp::ScheduleType_OMP::value());
}

// Regression test: a map that already carries a parallel (non-None) schedule must be left
// untouched -- the scheduler's find() returns SKIP instead of re-parallelizing it.
TEST(OMPSchedulerTest, SkipsAlreadyParallelMap) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc_2(base_desc);
    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto indvar = symbolic::symbol("i");
    auto& map = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        omp::ScheduleType_OMP::create()
    );
    auto& block = builder.add_block(map.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, desc_2);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    scheduling::loops::LoopSchedulingPass loop_scheduling_pass({get_omp_sched()}, nullptr);

    EXPECT_FALSE(loop_scheduling_pass.run(builder, analysis_manager));
    EXPECT_EQ(map.schedule_type().value(), omp::ScheduleType_OMP::value());
}

// Innermost sequential maps are marked for vectorization through the recorder; parallel maps are not.
TEST(OMPSchedulerTest, VectorizesInnermostSequentialMaps) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::Int64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("A", types::Pointer(), true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto& parallel = builder.add_map(
        root,
        j,
        symbolic::Lt(j, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(j, symbolic::one()),
        omp::ScheduleType_OMP::create()
    );
    auto& block = builder.add_block(parallel.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {j}, desc);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {j}, desc);

    auto& seq = builder.add_map(
        root,
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::zero(),
        symbolic::add(i, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block_2 = builder.add_block(seq.root());
    auto& b_in = builder.add_access(block_2, "A");
    auto& b_out = builder.add_access(block_2, "A");
    auto& tasklet_2 = builder.add_tasklet(block_2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block_2, b_in, tasklet_2, "_in", {i}, desc);
    builder.add_computational_memlet(block_2, tasklet_2, "_out", b_out, {i}, desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::scheduler::OMPScheduler scheduler;
    transformations::Recorder recorder;
    scheduler.set_recorder(&recorder);
    std::vector<structured_control_flow::StructuredLoop*> scheduled;
    EXPECT_TRUE(scheduler.post_schedule(builder, analysis_manager, scheduled));

    // Already-parallel maps keep their schedule; the sequential one is vectorized and recorded.
    EXPECT_FALSE(structured_control_flow::ScheduleType_Vectorize::is_set(parallel.schedule_type()));
    EXPECT_TRUE(structured_control_flow::ScheduleType_Vectorize::is_set(seq.schedule_type()));
    auto history = recorder.get_history();
    ASSERT_EQ(history.size(), 1);
    EXPECT_EQ(history[0]["transformation_type"], "VectorizeTransform");
}
