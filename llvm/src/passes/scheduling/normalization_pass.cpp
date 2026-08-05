#include "docc/passes/scheduling/normalization_pass.h"

#include <sdfg/passes/normalization/normalize.h>

namespace docc {
namespace passes {

llvm::PreservedAnalyses NormalizationPass::
    run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &AM) {
    auto &registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    registry.for_each_sdfg_modifiable(Module, [&](sdfg::StructuredSDFG &sdfg) {
        sdfg::passes::normalization::normalize(sdfg, false);
    });

    return llvm::PreservedAnalyses::all();
}


} // namespace passes
} // namespace docc
