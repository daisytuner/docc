#include "docc/passes/scheduling/scheduling_pass.h"

#include "docc/analysis/sdfg_registry.h"
#include "docc/cmd_args.h"
#include "docc/passes/dumps/pass_report_collector.h"
#include "sdfg/passes/rpc/daisytuner_rpc_context.h"
#include "sdfg/passes/rpc/rpc_scheduler.h"
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"

namespace docc {
namespace passes {

SchedulingPass::SchedulingPass(
    std::shared_ptr<sdfg::plugins::Context> context,
    const target::TargetOptions& target_options,
    bool force_synchronous,
    bool dump_visualization,
    bool transfer_opt
)
    : context_(std::move(context)), target_options_(target_options),
      force_synchronous_(force_synchronous || DOCC_FORCE_SYNCHRONOUS_OFFLOADING),
      dump_visualization_(dump_visualization), transfer_opt_(transfer_opt) {}

llvm::PreservedAnalyses SchedulingPass::
    run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    auto& report_collector = AM.get<docc::passes::PassReportManager>().get_collector(Module);

    auto& registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    auto offload_unknown_sizes = docc::DOCC_OFFLOAD_UNKNOWN_SIZES.getValue();

    std::unique_ptr<sdfg::passes::rpc::RPCScheduler> rpc;

    std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> schedulers;
    if (target_options_.remote_tuning) {
        std::shared_ptr<sdfg::passes::rpc::RpcContext> rpc_ctx =
            sdfg::passes::rpc::DaisytunerRpcContext::from_docc_config();
        rpc = std::make_unique<sdfg::passes::rpc::RPCScheduler>(
            rpc_ctx, target_options_.target, target_options_.category, dump_visualization_
        );
        schedulers.push_back(std::move(rpc));
    }

    auto docc_target = context_->get_target_handler(target_options_.target);
    if (docc_target) {
        auto target_schedulers = docc_target->safe_get_target_loop_schedulers(target_options_);
        if (!target_schedulers.empty()) {
            schedulers.insert(schedulers.end(), target_schedulers.begin(), target_schedulers.end());
        }
    }


    registry.for_each_sdfg_modifiable(Module, [&](analysis::SDFGHolder&, sdfg::StructuredSDFG& sdfg) {
        sdfg::builder::StructuredSDFGBuilder builder(sdfg);
        sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
        report_collector.in_scope(&builder.subject());

        sdfg::passes::scheduler::LoopSchedulingPass
            loop_scheduling_pass(schedulers, &report_collector, offload_unknown_sizes);
        loop_scheduling_pass.run(builder, analysis_manager);
    });

    report_collector.no_scope();
    return llvm::PreservedAnalyses::all();
}

} // namespace passes
} // namespace docc
