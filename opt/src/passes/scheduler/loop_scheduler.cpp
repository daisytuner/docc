#include "sdfg/passes/scheduler/loop_scheduler.h"

namespace sdfg {
namespace passes {
namespace scheduler {

bool LoopScheduler::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
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

    // Scheduling state machine
    bool applied = false;
    while (!queue.empty()) {
        auto loop = queue.front();
        queue.pop_front();

        auto scheduling_info = scheduling_info_map.at(loop);
        scheduling_info_map.erase(loop);

        SchedulerAction action;
        if (scheduling_info.loop_info.has_side_effects) {
            action = SchedulerAction::CHILDREN;
        } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(loop)) {
            action = schedule(builder, analysis_manager, *while_loop);
        } else if (auto structured_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            action = schedule(builder, analysis_manager, *structured_loop);
        } else {
            throw InvalidSDFGException("LoopScheduler encountered non-loop in loop analysis.");
        }

        switch (action) {
            case SchedulerAction::NEXT: {
                applied = true;
                break;
            }
            case SchedulerAction::CHILDREN: {
                auto children = loop_analysis.children(loop);
                if (children.empty()) {
                    continue;
                }
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

    return applied;
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
