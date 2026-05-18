#include "docc/passes/scheduling/scheduling_pass.h"

#include "docc/analysis/sdfg_registry.h"
#include "docc/cmd_args.h"
#include "sdfg/passes/scheduler/loop_scheduling_pass.h"

namespace docc {
namespace passes {

SchedulingPass::
    SchedulingPass(bool force_synchronous, bool dump_visualization, bool transfer_opt, sdfg::PassReportConsumer* report)
    : force_synchronous_(force_synchronous || DOCC_FORCE_SYNCHRONOUS_OFFLOADING),
      dump_visualization_(dump_visualization), transfer_opt_(transfer_opt), report_(report) {}

llvm::PreservedAnalyses SchedulingPass::
    run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    auto& registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    auto target = docc::DOCC_TUNE.getValue();
    auto remote_tuning = docc::DOCC_TRANSFERTUNE.getValue();
    auto offload_unknown_sizes = docc::DOCC_OFFLOAD_UNKNOWN_SIZES.getValue();

    std::vector<std::string> targets;
    if (target != "tenstorrent" && remote_tuning) {
        targets.push_back("rpc");
    }
    if (target != "sequential") {
        targets.push_back(target);
    }

    registry.for_each_sdfg_modifiable(Module, [&](analysis::SDFGHolder&, sdfg::StructuredSDFG& sdfg) {
        sdfg::builder::StructuredSDFGBuilder builder(sdfg);
        sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
        if (report_) report_->in_scope(&builder.subject());

        sdfg::passes::scheduler::LoopSchedulingPass loop_scheduling_pass(targets, report_, offload_unknown_sizes);
        loop_scheduling_pass.run(builder, analysis_manager);
    });

    if (report_) report_->no_scope();
    return llvm::PreservedAnalyses::all();
}

} // namespace passes
} // namespace docc
