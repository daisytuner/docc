#pragma once

#include <llvm/IR/Analysis.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>

#include "docc/analysis/analysis.h"
#include "docc/analysis/sdfg_registry.h"

namespace docc {
namespace passes {

class ArgumentExpansionPass : public llvm::PassInfoMixin<ArgumentExpansionPass> {
private:
    bool expand_arguments(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        const std::string& callee_name,
        const docc::analysis::Attributes& attributes,
        const std::string& target = "EXTERNAL"
    );

public:
    static bool available(analysis::AnalysisManager& AM) { return analysis::SDFGRegistry::is_link_time(AM); }

    llvm::PreservedAnalyses run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM);
};

} // namespace passes
} // namespace docc
