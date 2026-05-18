#pragma once

#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "docc/analysis/analysis.h"
#include "docc/passes/docc_pass.h"

namespace docc {
namespace passes {

class DumpSDFGPass : public llvm::PassInfoMixin<DumpSDFGPass> {
private:
    std::string stage_;
    bool dump_visualization_;

public:
    explicit DumpSDFGPass(std::string stage = "", bool dump_visualization = false)
        : stage_(std::move(stage)), dump_visualization_(dump_visualization) {}


    static bool available(analysis::AnalysisManager &AM) { return true; }

    llvm::PreservedAnalyses run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &AM);
};

} // namespace passes
} // namespace docc
