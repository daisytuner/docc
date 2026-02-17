#include "sdfg/passes/rpc/rpc_scheduler.h"
#include <memory>

#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/transformations/rpc_node_transform.h"

namespace sdfg {
namespace passes {
namespace rpc {

RPCScheduler::
    RPCScheduler(std::shared_ptr<rpc::RpcContext> rpc_context, std::string target, std::string category, bool print_steps)
    : LoopScheduler(), rpc_context_(std::move(rpc_context)), target_(std::move(target)), category_(std::move(category)),
      print_steps_(print_steps) {}

scheduler::SchedulerAction RPCScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);
    if (loop_info.loopnest_index == -1 || loop_info.has_side_effects || loop_info.is_elementwise) {
        return scheduler::NEXT;
    }

    transformations::RPCNodeTransform rpc_transform(loop, target_, category_, *rpc_context_);
    rpc_transform.set_report(report_);

    if (rpc_transform.can_be_applied(builder, analysis_manager)) {
        rpc_transform.apply(builder, analysis_manager);
        return scheduler::NEXT;
    }

    if (loop_info.num_maps <= 1) {
        return scheduler::NEXT;
    } else {
        // Visit 1st-level children
        return scheduler::CHILDREN;
    }
}

scheduler::SchedulerAction RPCScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);
    if (loop_info.loopnest_index == -1 || loop_info.has_side_effects) {
        return scheduler::NEXT;
    }
    else {
        return scheduler::CHILDREN;
    }

    // Apply transfer tuning to the loop
    transformations::RPCNodeTransform rpc_transform(loop, target_, category_, *rpc_context_, print_steps_);
    rpc_transform.set_report(report_);
    if (rpc_transform.can_be_applied(builder, analysis_manager)) {
        rpc_transform.apply(builder, analysis_manager);
        return scheduler::NEXT;
    }

    return scheduler::NEXT;
}

std::unordered_set<ScheduleTypeCategory> RPCScheduler::compatible_types() { return {ScheduleTypeCategory::None}; }

void register_rpc_loop_opt(
    std::shared_ptr<rpc::RpcContext> rpc_context,
    const std::string& target,
    const std::string& category,
    bool print_steps
) {
    scheduler::SchedulerRegistry::instance()
        .register_loop_scheduler<RPCScheduler>(RPCScheduler::target(), rpc_context, target, category, print_steps);
}

} // namespace rpc
} // namespace passes
} // namespace sdfg
