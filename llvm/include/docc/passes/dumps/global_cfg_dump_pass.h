#pragma once

#include "docc/analysis/analysis.h"
#include "docc/analysis/global_cfg_analysis.h"
#include "docc/passes/docc_pass.h"

namespace docc::passes {

class GlobalCFGPrinterPass : public llvm::PassInfoMixin<GlobalCFGPrinterPass> {
private:
    const std::string *out_path_;

public:
    GlobalCFGPrinterPass(const std::string *out_path = nullptr) : out_path_(out_path) {}

    static bool available(analysis::AnalysisManager &AM) {
        return analysis::AnalysisManager::available<analysis::GlobalCFGAnalysis>(AM);
    }

    llvm::PreservedAnalyses run(llvm::Module &Mod, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &GAM);

    static std::string escapeDot(std::string input);

    void dumpToConsole(llvm::Module &Mod, analysis::GlobalCFGAnalysis &cfg);

    void dumpToDotFile(const std::string &path, llvm::Module &Mod, analysis::GlobalCFGAnalysis &cfg);
};

} // namespace docc::passes
