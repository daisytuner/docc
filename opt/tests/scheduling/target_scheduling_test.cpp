#include "sdfg/scheduling/target_scheduling.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace sdfg;

namespace {

std::vector<std::string> steps;

class StepSDFGScheduler : public passes::scheduler::SDFGScheduler {
public:
    bool schedule(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&) override {
        steps.push_back("sdfg_scheduler");
        return false;
    }
};

class StepLoopScheduler : public passes::scheduler::LoopScheduler {
public:
    passes::scheduler::SchedulerAction find(
        builder::StructuredSDFGBuilder&, analysis::AnalysisManager&, structured_control_flow::StructuredLoop&, bool
    ) override {
        return passes::scheduler::NEXT;
    }
    passes::scheduler::SchedulerAction
    find(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&, structured_control_flow::While&, bool) override {
        return passes::scheduler::NEXT;
    }
    bool can_apply_schedule(
        builder::StructuredSDFGBuilder&, analysis::AnalysisManager&, structured_control_flow::StructuredLoop&, bool
    ) override {
        return false;
    }
    void apply_schedule(
        builder::StructuredSDFGBuilder&, analysis::AnalysisManager&, structured_control_flow::StructuredLoop&, bool
    ) override {
    }
    bool post_schedule(
        builder::StructuredSDFGBuilder&,
        analysis::AnalysisManager&,
        std::vector<structured_control_flow::StructuredLoop*>&
    ) override {
        steps.push_back("loop_scheduler");
        return false;
    }
    std::unordered_set<structured_control_flow::ScheduleTypeCategory> compatible_types() override {
        return {structured_control_flow::ScheduleTypeCategory::None};
    }
};

class StepRemoteTuning : public passes::Pass {
public:
    std::string name() override {
        return "StepRemoteTuning";
    }
    bool run_pass(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&) override {
        steps.push_back("remote_tuning");
        return false;
    }
};

docc::target::DoccTarget step_target = {
    .short_name = "steps",
    .apply_sched_time_mapping =
        [](builder::StructuredSDFGBuilder&, analysis::AnalysisManager&, const docc::target::TargetOptions&) -> bool {
        steps.push_back("mapping");
        return false;
    },
    .scheduling = {
        .create_sdfg_scheduler =
            [](const docc::target::TargetOptions&) -> std::unique_ptr<passes::scheduler::SDFGScheduler> {
            return std::make_unique<StepSDFGScheduler>();
        },
        .create_loop_schedulers =
            [](const docc::target::TargetOptions&) -> std::vector<std::shared_ptr<passes::scheduler::LoopScheduler>> {
            return {std::make_shared<StepLoopScheduler>()};
        }
    }
};

} // namespace

TEST(TargetSchedulingTest, RunsStepsInPipelineOrder) {
    auto context = plugins::Context::global_context();
    context.add_target(&step_target);
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());
    StepRemoteTuning remote_tuning;

    steps.clear();
    scheduling::schedule(context, builder, analysis_manager, {.target = "steps"}, {.remote_tuning = &remote_tuning});
    EXPECT_EQ(steps, (std::vector<std::string>{"sdfg_scheduler", "mapping", "remote_tuning", "loop_scheduler"}));

    steps.clear();
    scheduling::schedule(context, builder, analysis_manager, {.target = "steps"}, {.schedule_loops = false});
    EXPECT_EQ(steps, (std::vector<std::string>{"sdfg_scheduler", "mapping"}));
}

TEST(TargetSchedulingTest, WithoutNormalizationSkipsTheTargetsSDFGScheduler) {
    auto context = plugins::Context::global_context();
    context.add_target(&step_target);
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    steps.clear();
    scheduling::schedule(context, builder, analysis_manager, {.target = "steps"}, {.normalize = false});
    EXPECT_EQ(steps, (std::vector<std::string>{"mapping", "loop_scheduler"}));
}

TEST(TargetSchedulingTest, UnknownTargetThrows) {
    auto context = plugins::Context::global_context();
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    EXPECT_THROW(scheduling::schedule(context, builder, analysis_manager, {.target = "unknown"}), std::invalid_argument);
}

TEST(TargetSchedulingTest, TargetWithoutSchedulingIsRejected) {
    auto context = plugins::Context::global_context();
    docc::target::DoccTarget target = {.short_name = "incomplete"};

    EXPECT_THROW(context.add_target(&target), std::invalid_argument);
    EXPECT_EQ(context.get_target_handler("incomplete"), nullptr);
}
