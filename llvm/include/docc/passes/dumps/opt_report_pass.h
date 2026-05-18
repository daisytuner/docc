#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "docc/analysis/analysis.h"
#include "docc/passes/docc_pass.h"
#include "docc/passes/dumps/pass_report_collector.h"

namespace docc {
namespace passes {

class OPTReportPass : public llvm::PassInfoMixin<OPTReportPass> {
private:
    std::shared_ptr<docc::passes::PassReportCollector> report_;

public:
    explicit OPTReportPass(std::shared_ptr<docc::passes::PassReportCollector> report) : report_(std::move(report)) {}

    static bool available(analysis::AnalysisManager &AM) { return true; }

    llvm::PreservedAnalyses run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &AM);
};

} // namespace passes
} // namespace docc
