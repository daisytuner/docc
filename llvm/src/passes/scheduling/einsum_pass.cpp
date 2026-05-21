#include "docc/passes/scheduling/einsum_pass.h"

#include <llvm/IR/Analysis.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>

#include "docc/analysis/analysis.h"
#include "docc/analysis/sdfg_registry.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/passes/einsum.h"
#include "sdfg/passes/pipeline.h"

namespace docc {
namespace passes {

llvm::PreservedAnalyses EinsumPass::
    run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    auto& registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    registry.for_each_sdfg_modifiable(Module, [&](sdfg::StructuredSDFG& sdfg) {
        sdfg::builder::StructuredSDFGBuilder builder(sdfg);
        sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
        if (report_) report_->in_scope(&sdfg);

        // Run dataflow simplification pipeline, but ignore library nodes
        sdfg::passes::Pipeline dataflow_simplification = sdfg::passes::Pipeline::dataflow_simplification(true);
        dataflow_simplification.run(builder, analysis_manager);

        // Lift Einsum nodes to detect more library nodes (offloading)
        sdfg::passes::EinsumDetectionPass einsum_detection_pass;
        einsum_detection_pass.run(builder, analysis_manager);

        // Convert einsum into blas nodes (best-effort)
        sdfg::passes::EinsumConversionPass einsum_conversion_pass;
        einsum_conversion_pass.run(builder, analysis_manager);

        sdfg::passes::Pipeline lower("EinsumLower");
        lower.register_pass<sdfg::passes::EinsumLowerPass>();
        lower.run(builder, analysis_manager);

        sdfg::passes::Pipeline data_parallelism = sdfg::passes::Pipeline::data_parallelism();
        data_parallelism.run(builder, analysis_manager);
    });

    if (report_) report_->no_scope();

    return llvm::PreservedAnalyses::all();
}

} // namespace passes
} // namespace docc
