#include "docc/passes/scheduling/normalization_pass.h"

#include <sdfg/passes/normalization/normalization.h>

namespace docc {
namespace passes {

llvm::PreservedAnalyses NormalizationPass::
    run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &AM) {
    auto &registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    registry.for_each_sdfg_modifiable(Module, [&](sdfg::StructuredSDFG &sdfg) {
        sdfg::builder::StructuredSDFGBuilder builder(sdfg);
        sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

        sdfg::passes::Pipeline lp_pipeline = sdfg::passes::normalization::loop_normalization();
        lp_pipeline.run(builder, analysis_manager);
    });

    return llvm::PreservedAnalyses::all();
}


} // namespace passes
} // namespace docc
