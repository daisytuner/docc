#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "sdfg/tiles/tile_target.h"

namespace sdfg {
namespace tiles {

/**
 * @brief Maps a schedule `value()` to the @ref TileTarget that owns it.
 *
 * A target registers its @ref TileTarget under each schedule value it owns (e.g.
 * CUDA under `"CUDA"` and `"CUDA_Offload"`) in its `register_*_plugin`. The tile
 * algebra resolves the owner by `sched.value()`; unregistered schedules fall back
 * to the neutral rule in @ref AxisSchedule::classify. First registration wins.
 */
class TileTargetRegistry {
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<TileTarget>> targets_;

public:
    static TileTargetRegistry& instance() {
        static TileTargetRegistry registry;
        return registry;
    }

    void register_target(const std::string& schedule_value, std::shared_ptr<TileTarget> target) {
        std::lock_guard<std::mutex> lock(mutex_);
        targets_.emplace(schedule_value, std::move(target));
    }

    /// The target owning @p schedule_value, or nullptr if none is registered.
    const TileTarget* get(const std::string& schedule_value) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = targets_.find(schedule_value);
        return it == targets_.end() ? nullptr : it->second.get();
    }

    /// The implementation selector the target owning @p schedule_value stamps onto
    /// its tile codegen library nodes, or @ref data_flow::ImplementationType_NONE
    /// when no target is registered. Lets a transformation pick the right backend
    /// dispatcher without naming CUDA/ROCm.
    data_flow::ImplementationType implementation_type(const std::string& schedule_value) const {
        const TileTarget* target = get(schedule_value);
        return target ? target->implementation_type() : data_flow::ImplementationType_NONE;
    }

    /// Classify @p sched into cooperation facts via the owning target, or the
    /// neutral fallback when none is registered (`std::nullopt` for a sequential
    /// loop, else device-wide global cooperation without a scratchpad). This is
    /// the classification seam a caller reaches through its own context's registry.
    std::optional<AxisSchedule> classify(const structured_control_flow::ScheduleType& sched) const;

    /// The cooperation @ref Level for @p sched, or `std::nullopt` when no registered
    /// target treats it as a scratchpad (device-parallel) schedule.
    std::optional<Level> classify_level(const structured_control_flow::ScheduleType& sched) const;

    /// True when @p sched is a group-level schedule whose parallel threads can drive
    /// a cooperative shared-memory staging copy (a genuine offload schedule).
    bool drives_cooperative_copy(const structured_control_flow::ScheduleType& sched) const;
};

} // namespace tiles
} // namespace sdfg
