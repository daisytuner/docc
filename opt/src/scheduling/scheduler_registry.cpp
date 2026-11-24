#include "sdfg/scheduling/scheduler_registry.h"

namespace sdfg {
namespace scheduling {

std::unique_ptr<::sdfg::passes::scheduler::LoopScheduler> create_loop_scheduler(const std::string target) {
    auto scheduler = SchedulerRegistry::instance().get_loop_scheduler(target);
    if (scheduler) {
        return std::unique_ptr<::sdfg::passes::scheduler::LoopScheduler>(scheduler);
    }

    throw std::runtime_error("Unsupported scheduling target: " + target);
};


} // namespace scheduling
} // namespace sdfg
