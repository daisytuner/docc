#include "docc/passes/scheduling/scheduling_pass.h"

#include "docc/analysis/sdfg_registry.h"
#include "docc/cmd_args.h"
#include "docc/passes/dumps/dump_sdfg_pass.h"
#include "sdfg/passes/rpc/daisytuner_rpc_context.h"
#include "sdfg/passes/rpc/rpc_scheduling_pass.h"
#include "sdfg/scheduling/target_scheduling.h"

namespace docc {
namespace passes {

SchedulingPass::SchedulingPass(
    const sdfg::plugins::Context& context,
    bool force_synchronous,
    bool dump_visualization,
    bool transfer_opt,
    sdfg::PassReportConsumer* report
)
    : context_(context), force_synchronous_(force_synchronous || DOCC_FORCE_SYNCHRONOUS_OFFLOADING),
      dump_visualization_(dump_visualization), transfer_opt_(transfer_opt), report_(report) {
}

llvm::PreservedAnalyses SchedulingPass::
    run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    auto& registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    docc::target::TargetOptions options{
        .target = docc::DOCC_TUNE.getValue(),
        .category = docc::DOCC_TRANSFERTUNE_CATEGORY.getValue(),
        .remote_tuning = docc::DOCC_TRANSFERTUNE.getValue()
    };
    bool normalize = docc::DOCC_NORMALIZE.getValue();
    auto* target = context_.get_target_handler(options.target);
    if (target == nullptr) {
        throw std::invalid_argument("Unknown target '" + options.target + "'");
    }

    registry.for_each_sdfg_modifiable(Module, [&](analysis::SDFGHolder&, sdfg::StructuredSDFG& sdfg) {
        sdfg::builder::StructuredSDFGBuilder builder(sdfg);
        sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
        if (report_) {
            report_->in_scope(&builder.subject());
        }

        sdfg::scheduling::ScheduleConfig config{
            .normalize = normalize,
            .offload_unknown_sizes = docc::DOCC_OFFLOAD_UNKNOWN_SIZES.getValue(),
            .on_normal_form =
                [](const sdfg::StructuredSDFG& sdfg) {
                    DumpSDFGPass::dump(sdfg);
                },
            .report = report_
        };

        if (options.remote_tuning) {
            config.remote_tuning = [&options, this](bool remote_normalize) {
                return std::make_unique<sdfg::passes::scheduler::RpcOptimizationPass>(
                    sdfg::passes::rpc::DaisytunerRpcContext::from_docc_config(), options, remote_normalize, true, report_
                );
            };
        }

        sdfg::scheduling::schedule(context_, builder, analysis_manager, options, config);
    });

    if (report_) {
        report_->no_scope();
    }
    return llvm::PreservedAnalyses::all();
}

} // namespace passes
} // namespace docc
