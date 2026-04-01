#include "docc/target/tenstorrent/tenstorrent_scheduler.h"
#include "docc/target/tenstorrent/tenstorrent_transform.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction TenstorrentScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    if (auto map = dynamic_cast<structured_control_flow::Map*>(&loop)) {
        tenstorrent::TenstorrentTransform tt_transform(builder, analysis_manager, *map, false, offload_unknown_sizes);
        auto tt_plan = tt_transform.try_create_transform_plan(builder, analysis_manager);
        if (tt_plan != nullptr) {
            tt_transform.apply_plan(builder, analysis_manager, std::move(tt_plan));
            analysis_manager.invalidate_all();
            return NEXT;
        }
    }
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);

    // Check if in not outermost loop
    if (loop_info.loopnest_index == -1) {
        return NEXT;
    } else {
        // Visit 1st-level children
        return CHILDREN;
    }
}

SchedulerAction TenstorrentScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);
    // Check if in not outermost loop
    if (loop_info.loopnest_index == -1) {
        return NEXT;
    } else {
        // Visit 1st-level children
        return CHILDREN;
    }
}

std::unordered_set<ScheduleTypeCategory> TenstorrentScheduler::compatible_types() {
    return {ScheduleTypeCategory::None};
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
