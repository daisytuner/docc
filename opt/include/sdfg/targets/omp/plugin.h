#pragma once

#include <memory>
#include <optional>
#include <string>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/passes/scheduler/omp_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/omp/codegen/omp_map_dispatcher.h"
#include "sdfg/targets/omp/schedule.h"
#include "sdfg/tiles/tile_target_registry.h"

namespace sdfg {
namespace omp {

class OMPTileTarget : public tiles::TileTarget {
public:
    std::optional<tiles::AxisSchedule> classify(const structured_control_flow::ScheduleType& sched) const override {
        if (sched.category() == structured_control_flow::ScheduleTypeCategory::None) {
            return std::nullopt;
        }
        return tiles::AxisSchedule(tiles::Level::Device, space(tiles::Level::Device), has_scratchpad());
    }

    types::StorageType storage_type(tiles::Space space) const override {
        return space == tiles::Space::Register ? types::StorageType::CPU_Stack() : types::StorageType::CPU_Heap();
    }

    /// A host thread pool has no on-chip scratchpad: every level cooperates through
    /// global memory.
    tiles::Space space(tiles::Level) const override { return tiles::Space::Global; }

    bool supports_cooperative_staging(const structured_control_flow::ScheduleType&) const override { return false; }

    unsigned lane_width() const override { return 1; }
};

inline void register_omp_plugin() {
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_OMP::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<OMPMapDispatcher>(
                language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
            );
        }
    );

    passes::scheduler::SchedulerRegistry::instance()
        .register_loop_scheduler<passes::scheduler::OMPScheduler>(passes::scheduler::OMPScheduler::target());

    tiles::TileTargetRegistry::instance().register_target(ScheduleType_OMP::value(), std::make_shared<OMPTileTarget>());
}

} // namespace omp
} // namespace sdfg
