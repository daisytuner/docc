#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "docc/analysis/analysis.h"
#include "sdfg/optimization_report/pass_report_consumer.h"

namespace docc {
namespace passes {

class SchedulingPass : public llvm::PassInfoMixin<SchedulingPass> {
private:
    const bool dump_visualization_;
    const bool force_synchronous_;
    bool transfer_opt_;
    sdfg::PassReportConsumer *const report_ = nullptr;

public:
    SchedulingPass(
        bool force_synchronous = false,
        bool dump_visualization = false,
        bool transfer_opt = true,
        sdfg::PassReportConsumer *report = nullptr
    );

    static bool available(analysis::AnalysisManager &AM) { return true; }

    llvm::PreservedAnalyses run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &AM);
};

} // namespace passes
} // namespace docc
