#include "sdfg/passes/scheduler/cuda_scheduler.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/targets/cuda/cuda.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(CUDASchedulerTest, OuterParallelMapWithInnerMap) {
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

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"cuda"}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    EXPECT_EQ(loop.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(loop_2.schedule_type().value(), cuda::ScheduleType_CUDA::value());
}

TEST(CUDASchedulerTest, OuterSequentialForWith2DMap) {
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

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"cuda"}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    EXPECT_EQ(loop_2.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(loop_3.schedule_type().value(), cuda::ScheduleType_CUDA::value());
}

TEST(CUDASchedulerTest, OuterWhileWithInnerMaps) {
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

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"cuda"}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    EXPECT_EQ(loop_2.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(loop_3.schedule_type().value(), cuda::ScheduleType_CUDA::value());
}

// Regression test: When multiple outermost maps exist and some are already
// CUDA-scheduled, the compatibility filter must skip ALL of them.
TEST(CUDASchedulerTest, NoDoubleSchedulingOfAlreadyCUDAMaps) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc_2(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);

    // Create 5 outermost maps: map_a through map_e
    // Maps b and d are pre-scheduled as CUDA to simulate prior RPC results

    auto make_map = [&](const std::string& indvar_name,
                        structured_control_flow::ScheduleType schedule_type) -> structured_control_flow::Map& {
        builder.add_container(indvar_name, sym_desc);
        auto indvar = symbolic::symbol(indvar_name);
        auto& map = builder.add_map(
            root,
            indvar,
            symbolic::Lt(indvar, symbolic::symbol("N")),
            symbolic::integer(0),
            symbolic::add(indvar, symbolic::integer(1)),
            schedule_type
        );
        auto& body = map.root();
        auto& block = builder.add_block(body);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, desc_2);
        return map;
    };

    auto& map_a = make_map("a", structured_control_flow::ScheduleType_Sequential::create());
    auto& map_b = make_map("b", cuda::ScheduleType_CUDA::create());
    auto& map_c = make_map("c", structured_control_flow::ScheduleType_Sequential::create());
    auto& map_d = make_map("d", cuda::ScheduleType_CUDA::create());
    auto& map_e = make_map("e", structured_control_flow::ScheduleType_Sequential::create());

    // Verify preconditions
    EXPECT_EQ(map_a.schedule_type().category(), structured_control_flow::ScheduleTypeCategory::None);
    EXPECT_EQ(map_b.schedule_type().category(), structured_control_flow::ScheduleTypeCategory::Offloader);
    EXPECT_EQ(map_c.schedule_type().category(), structured_control_flow::ScheduleTypeCategory::None);
    EXPECT_EQ(map_d.schedule_type().category(), structured_control_flow::ScheduleTypeCategory::Offloader);
    EXPECT_EQ(map_e.schedule_type().category(), structured_control_flow::ScheduleTypeCategory::None);

    analysis::AnalysisManager analysis_manager(builder.subject());

    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"cuda"}, nullptr);
    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    // Maps a, c, e should now be CUDA-scheduled
    EXPECT_EQ(map_a.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(map_c.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(map_e.schedule_type().value(), cuda::ScheduleType_CUDA::value());

    // Maps b, d must still have exactly the CUDA schedule type — NOT double-offloaded.
    // With the old bug, maps b and d would pass through the compatibility filter
    // (because queue.size() shrank when earlier maps were removed) and get a second
    // CUDATransform applied, resulting in nested offloading (cudaMemcpy of device ptrs).
    EXPECT_EQ(map_b.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(map_d.schedule_type().value(), cuda::ScheduleType_CUDA::value());

    // Verify no containers with double-prefixed device names exist
    // (e.g. __daisy_cuda_0___daisy_cuda_821__ would indicate double-offloading)
    for (auto& container : builder.subject().containers()) {
        std::string name = container;
        size_t first = name.find("__daisy_cuda_");
        if (first != std::string::npos) {
            size_t second = name.find("__daisy_cuda_", first + 1);
            EXPECT_EQ(second, std::string::npos)
                << "Container " << name << " has double CUDA prefix, indicating double-offloading";
        }
    }
}

// Regression test: When running the CUDA scheduler after another scheduler
// has already CUDA-scheduled some maps, the analysis must reflect the updated
// schedule types. This simulates the {"rpc", "cuda"} pipeline.
TEST(CUDASchedulerTest, MultipleTargetsNoDoubleScheduling) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc_2(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);

    // Create two outermost maps
    auto& map_1 = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& body = map_1.root();
        auto& block = builder.add_block(body);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, desc_2);
    }

    auto& map_2 = builder.add_map(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& body = map_2.root();
        auto& block = builder.add_block(body);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("j")}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("j")}, desc_2);
    }

    // Manually set map_1 to CUDA to simulate what the first target (rpc) would do
    builder.update_schedule_type(map_1, cuda::ScheduleType_CUDA::create());

    EXPECT_EQ(map_1.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(map_2.schedule_type().value(), "SEQUENTIAL");

    // Run CUDA scheduler — it must see map_1 is already CUDA and skip it
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"cuda"}, nullptr);
    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    // map_2 should now be CUDA
    EXPECT_EQ(map_2.schedule_type().value(), cuda::ScheduleType_CUDA::value());

    // map_1 should still be CUDA (not double-offloaded)
    EXPECT_EQ(map_1.schedule_type().value(), cuda::ScheduleType_CUDA::value());

    // No double CUDA prefixes in containers
    for (auto& container : builder.subject().containers()) {
        std::string name = container;
        size_t first = name.find("__daisy_cuda_");
        if (first != std::string::npos) {
            size_t second = name.find("__daisy_cuda_", first + 1);
            EXPECT_EQ(second, std::string::npos)
                << "Container " << name << " has double CUDA prefix, indicating double-offloading";
        }
    }
}

// Regression test: CUDA scheduler must not double-schedule maps that are already
// CUDA-scheduled (e.g. after RPC scheduling applied CUDATransform).
TEST(CUDASchedulerTest, NoDoubleSchedulingOfCUDAMaps) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc_2(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("B", opaque_desc, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("j", sym_desc);
    builder.add_container("k", sym_desc);
    builder.add_container("l", sym_desc);
    builder.add_container("m", sym_desc);

    auto bound = symbolic::symbol("N");

    // Create 5 outermost maps: first 3 are already CUDA-scheduled, last 2 are sequential.
    // This pattern mimics RPC scheduling 3 maps then CUDA scheduler seeing all 5.

    // Map 1 (CUDA)
    auto& map1 = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), bound),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        cuda::ScheduleType_CUDA::create()
    );
    {
        auto& body = map1.root();
        auto& block = builder.add_block(body);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("i")}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("i")}, desc_2);
    }

    // Map 2 (CUDA)
    auto& map2 = builder.add_map(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), bound),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1)),
        cuda::ScheduleType_CUDA::create()
    );
    {
        auto& body = map2.root();
        auto& block = builder.add_block(body);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("j")}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("j")}, desc_2);
    }

    // Map 3 (CUDA)
    auto& map3 = builder.add_map(
        root,
        symbolic::symbol("k"),
        symbolic::Lt(symbolic::symbol("k"), bound),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
        cuda::ScheduleType_CUDA::create()
    );
    {
        auto& body = map3.root();
        auto& block = builder.add_block(body);
        auto& a_in = builder.add_access(block, "A");
        auto& a_out = builder.add_access(block, "A");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, a_in, tasklet, "_in", {symbolic::symbol("k")}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", a_out, {symbolic::symbol("k")}, desc_2);
    }

    // Map 4 (Sequential - should be scheduled by CUDA)
    auto& map4 = builder.add_map(
        root,
        symbolic::symbol("l"),
        symbolic::Lt(symbolic::symbol("l"), bound),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("l"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& body = map4.root();
        auto& block = builder.add_block(body);
        auto& b_in = builder.add_access(block, "B");
        auto& b_out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, b_in, tasklet, "_in", {symbolic::symbol("l")}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", b_out, {symbolic::symbol("l")}, desc_2);
    }

    // Map 5 (Sequential - should be scheduled by CUDA)
    auto& map5 = builder.add_map(
        root,
        symbolic::symbol("m"),
        symbolic::Lt(symbolic::symbol("m"), bound),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("m"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& body = map5.root();
        auto& block = builder.add_block(body);
        auto& b_in = builder.add_access(block, "B");
        auto& b_out = builder.add_access(block, "B");
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, b_in, tasklet, "_in", {symbolic::symbol("m")}, desc_2);
        builder.add_computational_memlet(block, tasklet, "_out", b_out, {symbolic::symbol("m")}, desc_2);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({"cuda"}, nullptr);

    EXPECT_TRUE(loop_scheduling_pass.run(builder, analysis_manager));

    // Maps 1-3 should remain CUDA (not double-scheduled)
    EXPECT_EQ(map1.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(map2.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(map3.schedule_type().value(), cuda::ScheduleType_CUDA::value());

    // Maps 4-5 should now be CUDA-scheduled
    EXPECT_EQ(map4.schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(map5.schedule_type().value(), cuda::ScheduleType_CUDA::value());
}
