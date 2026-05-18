#include "docc/passes/function_to_sdfg_pass.h"

#include <sdfg/serializer/json_serializer.h>

#include <memory>
#include <nlohmann/json.hpp>

#include "docc/analysis/sdfg_registry.h"
#include "docc/cmd_args.h"
#include "docc/docc_paths.h"
#include "docc/lifting/function_to_sdfg.h"
#include "docc/lifting/lift_report.h"
#include "docc/utils.h"
#include "sdfg/plugins/plugins.h"
#include "sdfg/structured_sdfg.h"

using json = nlohmann::json;

llvm::cl::opt<bool> DOCC_NoInlineFunctions(
    "docc-no-inline-functions",
    llvm::cl::desc("Disables inlining of functions during DOCC extraction"),
    llvm::cl::init(false)
);

namespace docc {
namespace passes {

llvm::PreservedAnalyses FunctionToSDFGPass::
    run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    if (!this->applies(Module)) {
        return llvm::PreservedAnalyses::all();
    }

    // Create cutout directories
    std::filesystem::path docc_dir = utils::get_docc_work_dir();
    std::filesystem::path module_dir = docc_dir / utils::hash_module_name(Module);
    std::filesystem::create_directories(module_dir);

    auto subcompile_meta = args::collect_subcompile_override_flags();

    // Add extraction directory as module flag
    llvm::LLVMContext& Context = Module.getContext();
    llvm::MDString* docc_dir_str = llvm::MDString::get(Context, docc_dir.string());
    Module.addModuleFlag(llvm::Module::Warning, "docc.extract.dir", docc_dir_str);

    // Attempt conversion of host functions to host SDFGs
    // Any SESE ("LLVM Region") of a host function is a potential sdfg
    auto& FAM = MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(Module).getManager();
    std::list<std::unique_ptr<sdfg::StructuredSDFG>> sdfgs;

    if (!DOCC_NoInlineFunctions.getValue()) {
        // 1st Pass: LinkonceODR functions only (references are updated)
        for (auto& func : Module) {
            if (func.isDeclaration()) {
                continue;
            }

            if (is_blacklisted(func, true)) {
                continue;
            }

            lifting::FunctionToSDFG converter(func, FAM, true);
            auto res = converter.run();
            if (res.empty()) {
                continue;
            }

            // Multiple SESE regions converted into SDFGs
            for (auto& sdfg : res) {
                if (!subcompile_meta.empty()) {
                    sdfg->add_metadata("compile_opts", subcompile_meta);
                }
                sdfgs.push_back(std::move(sdfg));
            }
        }
    }

    // 2nd Pass: Other functions
    for (auto& func : Module) {
        if (func.isDeclaration()) {
            continue;
        }

        if (func.hasName() && !func.hasLinkOnceODRLinkage()) {
            bool found = false;
            for (auto& plugin : this->plugin_registry_.plugins) {
                std::list<std::unique_ptr<sdfg::StructuredSDFG>> plugin_sdfgs =
                    plugin.sdfg_plugin.sdfg_lookup(func.getName().str());
                if (plugin_sdfgs.size() > 0) {
                    for (auto& plugin_sdfg : plugin_sdfgs) {
                        sdfgs.push_back(std::move(plugin_sdfg));
                    }
                    found = true;
                    LLVM_DEBUG_PRINTLN(
                        "Replaced function " << func.getName().str() << " with SDFGs from plugin "
                                             << plugin.sdfg_plugin.name
                    );
                    break;
                }
            }
            if (found) {
                // Prevent globals from disappearing
                // TODO: Write an own function for this without lifting LLVM IR to SDFGs
                lifting::FunctionToSDFG converter(func, FAM, false);
                converter.run();
                continue;
            }
        }

        if (is_blacklisted(func, false)) {
            continue;
        }

        lifting::FunctionToSDFG converter(func, FAM, false);
        auto res = converter.run();
        if (res.empty()) {
            continue;
        }

        // Multiple SESE regions converted into SDFGs
        for (auto& sdfg : res) {
            if (!subcompile_meta.empty()) {
                sdfg->add_metadata("compile_opts", subcompile_meta);
            }
            sdfgs.push_back(std::move(sdfg));
        }
    }
    if (sdfgs.empty()) {
        return llvm::PreservedAnalyses::none();
    }

    // Dump JSON
    lifting::LiftingReport::dump_report(module_dir / "lifting_report.json");

    // Add to SDFG registry
    auto& sdfg_registry = AM.get<analysis::SDFGRegistry>();
    sdfg_registry.insert(Module, std::move(sdfgs));

    // Initial dump of SDFGs
    sdfg_registry.dump_sdfgs(Module);

    return llvm::PreservedAnalyses::none();
};

bool FunctionToSDFGPass::applies(llvm::Module& Module) {
    std::string triple_str = Module.getTargetTriple();
    llvm::Triple triple(triple_str);
    switch (triple.getArch()) {
        case llvm::Triple::x86:
        case llvm::Triple::x86_64:
        case llvm::Triple::arm:
        case llvm::Triple::aarch64:
            return true;
        default:
            return false;
    }
}

bool FunctionToSDFGPass::is_blacklisted(llvm::Function& function, bool apply_on_linkonce_odr) const {
    if (lifting::FunctionToSDFG::is_blacklisted(function, apply_on_linkonce_odr)) {
        return true;
    }
    if (function.hasName() && func_blacklist_) {
        auto name = function.getName().str();
        if (std::regex_match(name, func_blacklist_.value())) {
            LLVM_DEBUG_PRINTLN("Skipping <" << name << ">, blacklisted by regex!");
            return true;
        }
    }
    return false;
}

FunctionToSDFGPass::FunctionToSDFGPass(const PluginRegistry& plugin_registry) : plugin_registry_(plugin_registry) {
    if (!DOCC_FUNC_BLACKLIST.empty()) {
        func_blacklist_ = std::regex(DOCC_FUNC_BLACKLIST);
    }
}

} // namespace passes
} // namespace docc
