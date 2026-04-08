#include "sdfg/passes/scheduler/loop_scheduling_pass.h"

#include "sdfg/passes/offloading/data_transfer_minimization_pass.h"
#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {
namespace scheduler {

bool LoopSchedulingPass::run_pass_target(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, const std::string& target
) {
    auto scheduler = SchedulerRegistry::instance().get_loop_scheduler(target);
    if (!scheduler) {
        throw std::runtime_error("Unsupported scheduling target: " + target);
    }
    scheduler->set_report(report_);

    // ===== Phase 1: Find all applicable loops =====
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& flop_analysis = analysis_manager.get<analysis::FlopAnalysis>();

    // Initialize queue with outermost loops
    std::list<structured_control_flow::ControlFlowNode*> queue;
    std::unordered_map<structured_control_flow::ControlFlowNode*, SchedulerLoopInfo> scheduling_info_map;
    for (auto& loop : loop_analysis.outermost_loops()) {
        queue.push_back(loop);

        SchedulerLoopInfo info;
        info.loop_info = loop_analysis.loop_info(loop);
        info.flop = flop_analysis.get(loop);
        scheduling_info_map[loop] = info;
    }
    if (queue.empty()) {
        return false;
    }

    // Filter by compatible types
    for (int i = 0; i < queue.size(); i++) {
        auto loop = queue.front();
        queue.pop_front();

        auto descendants = loop_analysis.descendants(loop);

        bool found_incompatible = false;
        for (auto descendant : descendants) {
            if (auto map_node = dynamic_cast<structured_control_flow::Map*>(descendant)) {
                auto compatible_schedules = scheduler->compatible_types();
                if (compatible_schedules.find(map_node->schedule_type().category()) == compatible_schedules.end()) {
                    found_incompatible = true;
                    break;
                }
            }
        }

        if (!found_incompatible) {
            queue.push_back(loop);
        }
    }

    // Traverse loop tree using scheduler's find() to collect applicable loops
    std::vector<structured_control_flow::StructuredLoop*> applicable_loops;
    while (!queue.empty()) {
        auto loop = queue.front();
        queue.pop_front();

        auto scheduling_info = scheduling_info_map.at(loop);
        scheduling_info_map.erase(loop);

        // Set the report context
        if (report_ && scheduling_info.loop_info.loopnest_index >= 0) {
            report_->in_outermost_loop(scheduling_info.loop_info.loopnest_index);
        }

        SchedulerAction action;
        if (auto while_loop = dynamic_cast<structured_control_flow::While*>(loop)) {
            action = scheduler->find(builder, analysis_manager, *while_loop, offload_unknown_sizes_);
        } else if (auto structured_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            action = scheduler->find(builder, analysis_manager, *structured_loop, offload_unknown_sizes_);
        } else {
            throw InvalidSDFGException("LoopScheduler encountered non-loop in loop analysis.");
        }

        switch (action) {
            case SchedulerAction::NEXT: {
                if (auto structured_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
                    applicable_loops.push_back(structured_loop);
                }
                break;
            }
            case SchedulerAction::CHILDREN: {
                auto children = loop_analysis.children(loop);
                for (auto& child : children) {
                    queue.push_front(child);

                    SchedulerLoopInfo info;
                    info.loop_info = loop_analysis.loop_info(child);
                    info.flop = flop_analysis.get(child);
                    scheduling_info_map[child] = info;
                }
                break;
            }
        }
    }

    if (applicable_loops.empty()) {
        return false;
    }

    // ===== Phase 2: Pre-schedule (collapse + cleanup) =====
    scheduler->pre_schedule(builder, analysis_manager, applicable_loops);

    // ===== Phase 3: Apply scheduling transforms =====
    // Phase 3a: Collect loops where transform can be applied
    std::vector<structured_control_flow::StructuredLoop*> schedulable_loops;
    for (auto* loop : applicable_loops) {
        if (scheduler->can_apply_schedule(builder, analysis_manager, *loop, offload_unknown_sizes_)) {
            schedulable_loops.push_back(loop);
        }
    }

    if (schedulable_loops.empty()) {
        return false;
    }

    // Phase 3b: Apply transforms
    for (auto* loop : schedulable_loops) {
        scheduler->apply_schedule(builder, analysis_manager, *loop, offload_unknown_sizes_);
    }
    analysis_manager.invalidate_all();

    // ===== Phase 4: Post-schedule =====
    scheduler->post_schedule(builder, analysis_manager, schedulable_loops);

    return true;
}

bool LoopSchedulingPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (targets_.empty()) {
        return false;
    }
    if (targets_.size() == 1 && targets_[0] == "none") {
        return false;
    }

    bool applied = false;
    for (const auto& target : targets_) {
        bool target_applied = run_pass_target(builder, analysis_manager, target);
        applied = applied || target_applied;
    }

    if (applied) {
        DataTransferMinimizationPass dtm_pass;
        dtm_pass.run(builder, analysis_manager);
    }

    return applied;
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
