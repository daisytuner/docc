#include "docc/target/axelera/axelera_scheduler.h"

#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/transformations/loop_tiling.h>

#include "docc/target/axelera/datapath.h"
#include "docc/target/axelera/schedule.h"

namespace sdfg {
namespace passes {
namespace scheduler {

namespace {

std::optional<int64_t> constant(const symbolic::Expression& expr) {
    if (expr.is_null() || !SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return std::nullopt;
    }
    return SymEngine::down_cast<const SymEngine::Integer&>(*expr).as_int();
}

} // namespace

SchedulerAction AxeleraScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    if (dyn_cast<structured_control_flow::Map*>(&loop)) {
        return APPLY;
    }
    return CHILDREN;
}

SchedulerAction AxeleraScheduler::find(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    return NEXT;
}

bool AxeleraScheduler::can_apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto map = dyn_cast<structured_control_flow::Map*>(&loop);
    if (!map || !analysis_manager.get<analysis::LoopAnalysis>().children(&loop).empty()) {
        return false;
    }
    auto count = constant(loop.num_iterations());
    if (constant(loop.init()) != 0 || constant(loop.stride()) != 1 || !count || *count <= 0 ||
        *count % axelera::PWORD_LANES != 0) {
        return false;
    }

    try {
        auto plan = axelera::plan_datapath(builder.subject(), map->root(), map->indvar());
        auto streams = plan.inputs;
        streams.push_back(plan.output);
        for (auto& stream : streams) {
            auto first = constant(symbolic::expand(symbolic::subs(stream.index, map->indvar(), symbolic::zero())));
            if (!first || *first % axelera::PWORD_LANES != 0) {
                return false;
            }
        }
    } catch (const std::runtime_error&) {
        return false;
    }

    transformations::LoopTiling tiling(loop, axelera::PWORD_LANES);
    return tiling.can_be_applied(builder, analysis_manager);
}

void AxeleraScheduler::apply_schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    transformations::LoopTiling tiling(loop, axelera::PWORD_LANES);
    tiling.apply(builder, analysis_manager);
    builder.update_schedule_type(*tiling.outer_loop(), axelera::ScheduleType_Axelera_Datapath::create());
    builder.update_schedule_type(*tiling.inner_loop(), axelera::ScheduleType_Axelera_Lane::create());
}

std::unordered_set<ScheduleTypeCategory> AxeleraScheduler::compatible_types() {
    return {ScheduleTypeCategory::None};
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
