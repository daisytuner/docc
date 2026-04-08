#include "docc/target/tenstorrent/tenstorrent_scheduler.h"
#include "docc/target/tenstorrent/tenstorrent_transform.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction TenstorrentScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    if (dynamic_cast<structured_control_flow::Map*>(&loop)) {
        return NEXT;
    }

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);

    if (loop_info.loopnest_index == -1) {
        return NEXT;
    } else {
        return CHILDREN;
    }
}

SchedulerAction TenstorrentScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);
    if (loop_info.loopnest_index == -1) {
        return NEXT;
    } else {
        return CHILDREN;
    }
}

bool TenstorrentScheduler::can_apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto* map = dynamic_cast<structured_control_flow::Map*>(&loop);
    if (!map) {
        return false;
    }
    tenstorrent::TenstorrentTransform tt_transform(builder, analysis_manager, *map, false, offload_unknown_sizes);
    auto tt_plan = tt_transform.try_create_transform_plan(builder, analysis_manager);
    return tt_plan != nullptr;
}

void TenstorrentScheduler::apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto* map = dynamic_cast<structured_control_flow::Map*>(&loop);
    tenstorrent::TenstorrentTransform tt_transform(builder, analysis_manager, *map, false, offload_unknown_sizes);
    auto tt_plan = tt_transform.try_create_transform_plan(builder, analysis_manager);
    tt_transform.apply_plan(builder, analysis_manager, std::move(tt_plan));
}

std::unordered_set<ScheduleTypeCategory> TenstorrentScheduler::compatible_types() {
    return {ScheduleTypeCategory::None};
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
