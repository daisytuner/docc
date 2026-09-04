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
};

} // namespace tiles
} // namespace sdfg
