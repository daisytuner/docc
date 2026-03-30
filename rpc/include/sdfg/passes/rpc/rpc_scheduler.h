#pragma once

#include <memory>
#include <unordered_set>
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {
namespace rpc {

class RPCScheduler : public scheduler::LoopScheduler {
private:
    std::shared_ptr<rpc::RpcContext> rpc_context_;
    const std::string target_;
    const std::string category_;
    const bool print_steps_;

public:
    scheduler::SchedulerAction find(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) override;

    scheduler::SchedulerAction find(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::While& loop,
        bool offload_unknown_sizes = false
    ) override;

    bool can_apply_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) override;

    void apply_schedule(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::StructuredLoop& loop,
        bool offload_unknown_sizes = false
    ) override;

    RPCScheduler(
        std::shared_ptr<rpc::RpcContext> rpc_context, std::string target, std::string category, bool print_steps = false
    );

    static std::string target() { return "rpc"; }

    std::unordered_set<ScheduleTypeCategory> compatible_types() override;
};

void register_rpc_loop_opt(
    std::shared_ptr<rpc::RpcContext> rpc_context,
    const std::string& target,
    const std::string& category,
    bool print_steps = false
);

} // namespace rpc
} // namespace passes
} // namespace sdfg
