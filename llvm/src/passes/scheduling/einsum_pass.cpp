#include "docc/passes/scheduling/einsum_pass.h"

#include <llvm/IR/Analysis.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>

#include <stdexcept>

#include "docc/analysis/analysis.h"
#include "docc/analysis/sdfg_registry.h"
#include "docc/cmd_args.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/passes/einsum.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/passes/structured_control_flow/block_fusion.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

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

        sdfg::passes::Pipeline lift_and_extend("EinsumLiftAndExtend");
        lift_and_extend.register_pass<sdfg::passes::EinsumLiftPass>();
        lift_and_extend.register_pass<sdfg::passes::EinsumExtendPass>();
        lift_and_extend.register_pass<sdfg::passes::BlockFusionPass>();
        lift_and_extend.run(builder, analysis_manager);

        sdfg::passes::Pipeline expand("EinsumExpand");
        expand.register_pass<sdfg::passes::EinsumExpandPass>();
        expand.register_pass<sdfg::passes::DeadCFGElimination>();
        // LoopDistribute ?
        expand.run(builder, analysis_manager);

        bool applied;
        do {
            sdfg::passes::EinsumConversion conversion(builder, analysis_manager, report_);
            applied = conversion.visit();
            analysis_manager.invalidate_all();
        } while (applied);

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
