#include "sdfg/passes/scheduler/rocm_scheduler.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/targets/rocm/rocm.h"

#include <gtest/gtest.h>

using namespace sdfg;

static passes::scheduler::ROCMScheduler* get_rocm_sched() {
    static passes::scheduler::ROCMScheduler instance;
    return &instance;
}

// Regression test (remote-tuning double-offload): a lone top-level map that already carries a
// ROCM schedule has no nested maps, so it escapes the descendant-only compatibility filter. The
// scheduler must still leave it untouched (find() returns SKIP) instead of re-applying
// ROCMTransform, which would nest device buffers.
TEST(ROCMSchedulerTest, SkipsLoneAlreadyROCMScheduledMap) {
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

    auto indvar = symbolic::symbol("i");
    auto& map = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(indvar, symbolic::integer(1)),
        rocm::ScheduleType_ROCM::create()
    );
    auto& block = builder.add_block(map.root());
    auto& a_in = builder.add_access(block, "A");
    auto& a_out = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, a_in, tasklet, "_in", {indvar}, desc_2);
    builder.add_computational_memlet(block, tasklet, "_out", a_out, {indvar}, desc_2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::scheduler::LoopSchedulingPass loop_scheduling_pass({get_rocm_sched()}, nullptr);

    // The only loop is already scheduled -> no scheduling changes.
    EXPECT_FALSE(loop_scheduling_pass.run(builder, analysis_manager));

    // Schedule unchanged and no device buffers created (not re-offloaded).
    EXPECT_EQ(map.schedule_type().value(), rocm::ScheduleType_ROCM::value());
    for (auto& container : builder.subject().containers()) {
        std::string name = container;
        EXPECT_EQ(name.find(rocm::ROCM_DEVICE_PREFIX), std::string::npos)
            << "Unexpected device buffer " << name << " (map was re-offloaded)";
    }
}
