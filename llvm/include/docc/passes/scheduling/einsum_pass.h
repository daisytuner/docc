#pragma once

#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "docc/analysis/analysis.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/optimization_report/pass_report_consumer.h"

namespace docc {
namespace passes {

class EinsumPass : public llvm::PassInfoMixin<EinsumPass> {
private:
    sdfg::PassReportConsumer *report_;

public:
    EinsumPass(sdfg::PassReportConsumer *report = nullptr) : report_(report) {}

    static bool available(analysis::AnalysisManager &AM) { return true; }

    llvm::PreservedAnalyses run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &AM);
};

} // namespace passes
} // namespace docc
