#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "docc/analysis/analysis.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/plugins/plugins.h"

namespace docc {
namespace passes {

class SchedulingPass : public llvm::PassInfoMixin<SchedulingPass> {
private:
    std::shared_ptr<sdfg::plugins::Context> context_;
    target::TargetOptions target_options_;
    const bool dump_visualization_;
    const bool force_synchronous_;
    bool transfer_opt_;

public:
    SchedulingPass(
        std::shared_ptr<sdfg::plugins::Context> context,
        const target::TargetOptions &target_options,
        bool force_synchronous = false,
        bool dump_visualization = false,
        bool transfer_opt = true
    );

    static bool available(analysis::AnalysisManager &AM) { return true; }

    llvm::PreservedAnalyses run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &AM);
};

} // namespace passes
} // namespace docc
