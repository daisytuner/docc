#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <memory>
#include <thread>

#include "docc/analysis/analysis.h"
#include "docc/analysis/sdfg_registry.h"

namespace docc {
namespace passes {

template<typename PassT>
class DOCCPass : public llvm::PassInfoMixin<DOCCPass<PassT>> {
public:
    DOCCPass(PassT Impl, analysis::AnalysisManager &AM) : Impl_(std::move(Impl)), AM_(AM) {}

    template<typename... Ts>
    llvm::PreservedAnalyses run(llvm::Module &Mod, llvm::ModuleAnalysisManager &MAM) {
        if (!PassT::available(AM_)) {
            return llvm::PreservedAnalyses::all();
        }

        return Impl_.run(Mod, MAM, AM_);
    }

private:
    PassT Impl_;
    analysis::AnalysisManager &AM_;
};

template<typename PassT>
DOCCPass<PassT> createDOCCPass(PassT Pass, analysis::AnalysisManager &AM) {
    return DOCCPass<PassT>(std::move(Pass), AM);
}

} // namespace passes
} // namespace docc
