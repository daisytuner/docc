#include "docc/passes/sdfg_simplify_pass.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/passes/pipeline.h>

#include "docc/analysis/sdfg_registry.h"
#include "docc/utils.h"

namespace docc {
namespace passes {

llvm::PreservedAnalyses SDFGSimplifyPass::
    run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &AM) {
    auto &registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    registry.for_each_sdfg_modifiable(Module, [&](sdfg::StructuredSDFG &sdfg) {
        sdfg::passes::Pipeline expression_combine = sdfg::passes::Pipeline::expression_combine();
        sdfg::passes::Pipeline memlet_combine = sdfg::passes::Pipeline::memlet_combine();
        sdfg::passes::Pipeline controlflow_simplification = sdfg::passes::Pipeline::controlflow_simplification();
        sdfg::passes::Pipeline data_parallelism = sdfg::passes::Pipeline::data_parallelism();

        sdfg::builder::StructuredSDFGBuilder builder(sdfg);
        sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

        // Simplification passes
        expression_combine.run(builder, analysis_manager);
        memlet_combine.run(builder, analysis_manager);
        controlflow_simplification.run(builder, analysis_manager);

        // Data parallelism passes
        data_parallelism.run(builder, analysis_manager);

        // Simplification passes
        expression_combine.run(builder, analysis_manager);
        memlet_combine.run(builder, analysis_manager);
        controlflow_simplification.run(builder, analysis_manager);
    });

    return llvm::PreservedAnalyses::all();
}

} // namespace passes
} // namespace docc
