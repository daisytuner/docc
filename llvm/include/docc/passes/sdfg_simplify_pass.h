#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include "docc/analysis/analysis.h"

namespace docc {
namespace passes {

class SDFGSimplifyPass : public llvm::PassInfoMixin<SDFGSimplifyPass> {
public:
    static bool available(analysis::AnalysisManager &WPAM) { return true; }

    llvm::PreservedAnalyses run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &WPAM);
};

} // namespace passes
} // namespace docc
