
#include "sdfg/targets/memory/plugin.h"

#include <dlfcn.h>
#include <docc/target/tenstorrent/plugin.h>
#include <llvm/Analysis/GlobalsModRef.h>
#include <llvm/Analysis/ProfileSummaryInfo.h>
#include <llvm/Analysis/RegionInfo.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/DeadArgumentElimination.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/IndVarSimplify.h>
#include <llvm/Transforms/Scalar/LICM.h>
#include <llvm/Transforms/Scalar/LoopRotation.h>
#include <llvm/Transforms/Scalar/LoopSimplifyCFG.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimpleLoopUnswitch.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/TailRecursionElimination.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Utils/FixIrreducible.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
#include <llvm/Transforms/Utils/LowerInvoke.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/optimization_report/optimization_report.h>
#include <sdfg/passes/rpc/rpc_scheduler.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/targets/cuda/plugin.h>

#include <cstdlib>
#include <memory>
#include <mutex>

#include "docc/cmd_args.h"
#include "docc/docc.h"
#include "docc/passes/code_generation/code_generation_pass.h"
#include "docc/passes/docc_pass.h"
#include "docc/passes/dumps/dump_attributes_pass.h"
#include "docc/passes/dumps/dump_sdfg_pass.h"
#include "docc/passes/dumps/global_cfg_dump_pass.h"
#include "docc/passes/dumps/opt_report_pass.h"
#include "docc/passes/dumps/pass_report_collector.h"
#include "docc/passes/function_to_sdfg_pass.h"
#include "docc/passes/inlining/argument_expansion_pass.h"
#include "docc/passes/scheduling/docc_backend_context.h"
#include "docc/passes/scheduling/einsum_pass.h"
#include "docc/passes/scheduling/normalization_pass.h"
#include "docc/passes/scheduling/scheduling_pass.h"
#include "docc/passes/sdfg_simplify_pass.h"
#include "docc/plugin_registry.h"

llvm::cl::opt<bool> DOCC_LowerInvoke(
    "docc-lower-invoke", llvm::cl::desc("Lowers invoke instructions before DOCC processing"), llvm::cl::init(false)
);

llvm::cl::opt<bool> DOCC_Einsum("docc-einsum", llvm::cl::desc("Enables lifting Einstein notation."), llvm::cl::init(false));

llvm::cl::opt<std::string> DOCC_DUMP_SDFG(
    "docc-dump-sdfg",
    llvm::cl::desc("Enables Output of sdfgs"),
    llvm::cl::init("none"),
    llvm::cl::value_desc("none|normalization")
);

static docc::analysis::AnalysisManager AM;

#ifdef NDEBUG
inline constexpr bool debug_print = false;
#else
inline constexpr bool debug_print = true;
#endif

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        .APIVersion = LLVM_PLUGIN_API_VERSION,
        .PluginName = "DOCC",
        .PluginVersion = "v0.0.1",
        .RegisterPassBuilderCallbacks =
            [](llvm::PassBuilder &PB) {
                docc::register_sdfg_dispatchers();

                auto target = docc::DOCC_TUNE.getValue();
                auto category = docc::DOCC_TRANSFERTUNE_CATEGORY.getValue();

                auto remote_tuning = docc::DOCC_TRANSFERTUNE.getValue();

                if (remote_tuning) {
                    std::shared_ptr<sdfg::passes::rpc::RpcContext> context =
                        sdfg::passes::rpc::DaisytunerRpcContext::from_docc_config();
                    sdfg::passes::rpc::register_rpc_loop_opt(context, target, category);
                }

                // Compile-Time Pass Registration
                PB.registerPipelineStartEPCallback([](llvm::ModulePassManager &MPM, llvm::OptimizationLevel Level) {
                    // Simplification
                    {
                        llvm::FunctionPassManager FPM;
                        FPM.addPass(llvm::PromotePass());
                        FPM.addPass(llvm::EarlyCSEPass(true));
                        FPM.addPass(llvm::InstCombinePass());
                        FPM.addPass(llvm::SimplifyCFGPass());
                        FPM.addPass(llvm::TailCallElimPass());
                        FPM.addPass(llvm::SimplifyCFGPass());
                        FPM.addPass(llvm::ReassociatePass());
                        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
                    }

                    // Inlining
                    {
                        llvm::ModuleInlinerWrapperPass MIWP(llvm::getInlineParams());
                        MIWP.addModulePass(llvm::RequireAnalysisPass<llvm::GlobalsAA, llvm::Module>());
                        auto aa = llvm::InvalidateAnalysisPass<llvm::AAManager>();
                        MIWP.addModulePass(llvm::createModuleToFunctionPassAdaptor(std::move(aa)));
                        MIWP.addModulePass(llvm::RequireAnalysisPass<llvm::ProfileSummaryAnalysis, llvm::Module>());
                        llvm::CGSCCPassManager &MainCGPipeline = MIWP.getPM();
                        MainCGPipeline.addPass(llvm::PostOrderFunctionAttrsPass());

                        MPM.addPass(std::move(MIWP));
                    }

                    // Simplification
                    {
                        llvm::FunctionPassManager FPM;
                        FPM.addPass(llvm::PromotePass());
                        FPM.addPass(llvm::EarlyCSEPass(true));
                        FPM.addPass(llvm::InstCombinePass());
                        FPM.addPass(llvm::SimplifyCFGPass());
                        FPM.addPass(llvm::TailCallElimPass());
                        FPM.addPass(llvm::SimplifyCFGPass());
                        FPM.addPass(llvm::ReassociatePass());
                        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
                    }

                    // Loop rotation
                    {
                        llvm::LoopPassManager LPM;
                        LPM.addPass(llvm::LoopSimplifyCFGPass());
                        LPM.addPass(llvm::SimpleLoopUnswitchPass(true));

                        llvm::FunctionPassManager FPM;
                        FPM.addPass(llvm::createFunctionToLoopPassAdaptor<
                                    llvm::LoopPassManager>(std::move(LPM), false, false));
                        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
                    }

                    // Loop simplification
                    {
                        llvm::FunctionPassManager FPM;

                        llvm::LoopPassManager LPM;
                        LPM.addPass(llvm::IndVarSimplifyPass());
                        FPM.addPass(llvm::createFunctionToLoopPassAdaptor<
                                    llvm::LoopPassManager>(std::move(LPM), false, true));
                        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
                    }

                    // Simplification
                    {
                        llvm::FunctionPassManager FPM;
                        FPM.addPass(llvm::PromotePass());
                        FPM.addPass(llvm::EarlyCSEPass(true));
                        FPM.addPass(llvm::InstCombinePass());
                        FPM.addPass(llvm::SimplifyCFGPass());
                        FPM.addPass(llvm::TailCallElimPass());
                        FPM.addPass(llvm::SimplifyCFGPass());
                        FPM.addPass(llvm::ReassociatePass());
                        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
                    }

                    // CFG Simplifications
                    {
                        llvm::FunctionPassManager FPM;
                        if (DOCC_LowerInvoke.getValue()) {
                            FPM.addPass(llvm::LowerInvokePass());
                            FPM.addPass(llvm::SimplifyCFGPass());
                        }
                        FPM.addPass(llvm::LowerSwitchPass());
                        FPM.addPass(llvm::LoopSimplifyPass());
                        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
                    }

                    std::shared_ptr<docc::passes::PassReportCollector> report_consumer;
                    bool generate_opt_report_file = true;

                    report_consumer = std::make_shared<docc::passes::PassReportCollector>();

                    // Lift SDFGs from functions
                    MPM.addPass(docc::passes::createDOCCPass(docc::passes::FunctionToSDFGPass(docc::plugin_registry), AM)
                    );

                    // Normalization Pass
                    if (docc::DOCC_TUNE != "none") {
                        MPM.addPass(docc::passes::createDOCCPass(docc::passes::NormalizationPass(), AM));
                    }

                    // Einsum Pass
                    if (DOCC_Einsum) {
                        MPM.addPass(docc::passes::createDOCCPass(docc::passes::EinsumPass(), AM));
                        MPM.addPass(docc::passes::createDOCCPass(docc::passes::NormalizationPass(), AM));
                    }

                    bool enable_offloading_transfer_opt = !docc::args::DOCC_NO_OFFLOADING_TRANSFER_OPT.getValue();

                    if (docc::DOCC_TUNE != "none") {
                        // Dump sdfg and features after all seuquential optimizations are done but
                        // before offloading
                        MPM.addPass(docc::passes::createDOCCPass(docc::passes::DumpSDFGPass(), AM));
                        MPM.addPass(docc::passes::createDOCCPass(
                            docc::passes::
                                SchedulingPass(false, false, enable_offloading_transfer_opt, report_consumer.get()),
                            AM
                        ));
                    }

                    if (generate_opt_report_file) {
                        MPM.addPass(docc::passes::createDOCCPass(docc::passes::OPTReportPass(report_consumer), AM));
                    }

                    MPM.addPass(docc::passes::createDOCCPass(
                        docc::passes::DumpSDFGPass("scheduled", docc::args::DOCC_DOT_DUMP_SCHEDULED.getValue()), AM
                    ));
                    if (enable_offloading_transfer_opt) {
                        MPM.addPass(docc::passes::createDOCCPass(docc::passes::DumpAttributesPass(), AM));
                    }
                });

                // Link-Time Pass Registration (Early and Late)
                PB.registerOptimizerLastEPCallback([](llvm::ModulePassManager &MPM, llvm::OptimizationLevel Level) {
                    // Minimize data transfers
                    MPM.addPass(docc::passes::createDOCCPass(docc::passes::ArgumentExpansionPass(), AM));

                    if (docc::DOCC_DUMP_GLBL_CFG_EN) {
                        std::string *path_opt = docc::DOCC_DUMP_GLBL_CFG.getValue().empty()
                                                    ? nullptr
                                                    : &docc::DOCC_DUMP_GLBL_CFG.getValue();
                        MPM.addPass(docc::passes::createDOCCPass(docc::passes::GlobalCFGPrinterPass(path_opt), AM));
                    }

                    // Code Generation Passes
                    MPM.addPass(docc::passes::createDOCCPass(docc::passes::CodeGenerationPass(), AM));
                });
            }
    };
}
