#pragma once

#include <memory>
#include "sdfg/passes/scheduler/loop_scheduler.h"

namespace sdfg {
namespace passes {
namespace scheduler {

/**
 * @deprecated Global state and superfluous, do not use
 */
class SchedulerRegistry {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<LoopScheduler>> scheduler_map_;

public:
    static SchedulerRegistry& instance() {
        static SchedulerRegistry registry;
        return registry;
    }

    template<typename T, typename... Args>
    void register_loop_scheduler(std::string target, Args... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (scheduler_map_.find(target) != scheduler_map_.end()) {
            return;
        }
        scheduler_map_[target] = std::make_shared<T>(std::forward<Args>(args)...);
    }
    std::shared_ptr<LoopScheduler> get_loop_scheduler(std::string target) const {
        auto it = scheduler_map_.find(target);
        if (it != scheduler_map_.end()) {
            return it->second;
        }
        return nullptr;
    }

    size_t size_loop_schedulers() const { return scheduler_map_.size(); }
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
