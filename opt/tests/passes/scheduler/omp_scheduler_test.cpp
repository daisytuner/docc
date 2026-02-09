#include "sdfg/passes/scheduler/omp_scheduler.h"

#include "sdfg/passes/scheduler/loop_scheduling_pass.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/omp/schedule.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(OMPSchedulerTest, OuterParallelMapWithInnerMap) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"openmp"}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    EXPECT_EQ(loop.schedule_type().value(), omp::ScheduleType_OMP::value());
    EXPECT_EQ(loop_2.schedule_type().value(), structured_control_flow::ScheduleType_Sequential::value());
}

TEST(OMPSchedulerTest, OuterSequentialForWithInnerMaps) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");

    auto& loop = builder.add_for(
        root, indvar, symbolic::Lt(indvar, bound), symbolic::integer(0), symbolic::add(indvar, symbolic::integer(1))
    );
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
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar, indvar_2}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar, indvar_2}, desc_2);
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
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar, indvar_3}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar, indvar_3}, desc_2);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"openmp"}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    EXPECT_EQ(loop_2.schedule_type().value(), omp::ScheduleType_OMP::value());
    EXPECT_EQ(loop_3.schedule_type().value(), omp::ScheduleType_OMP::value());
}

TEST(OMPSchedulerTest, OuterSequentialForWith2DMap) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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
    auto bound = symbolic::symbol("N");
    auto indvar = symbolic::symbol("i");

    auto& loop = builder.add_for(
        root, indvar, symbolic::Lt(indvar, bound), symbolic::integer(0), symbolic::add(indvar, symbolic::integer(1))
    );
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

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"openmp"}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    EXPECT_EQ(loop_2.schedule_type().value(), omp::ScheduleType_OMP::value());
    EXPECT_EQ(loop_3.schedule_type().value(), structured_control_flow::ScheduleType_Sequential::value());
}

TEST(OMPSchedulerTest, OuterWhileWithInnerMaps) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
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

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"openmp"}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    EXPECT_EQ(loop_2.schedule_type().value(), omp::ScheduleType_OMP::value());
    EXPECT_EQ(loop_3.schedule_type().value(), omp::ScheduleType_OMP::value());
}
