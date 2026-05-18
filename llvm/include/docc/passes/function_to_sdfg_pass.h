#pragma once

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <filesystem>
#include <fstream>
#include <memory>

#include <sdfg/structured_sdfg.h>

#include "docc/analysis/analysis.h"
#include "docc/plugin_registry.h"
#include "docc/utils.h"

namespace docc {
namespace passes {

inline constexpr std::string_view SDFG_INDEX_JSON_FILE_NAME = "JSON";

class FunctionToSDFGPass : public llvm::PassInfoMixin<FunctionToSDFGPass> {
private:
    const PluginRegistry& plugin_registry_;
    std::optional<std::regex> func_blacklist_;

    bool applies(llvm::Module& Module);

    bool is_blacklisted(llvm::Function& function, bool apply_on_linkonce_odr) const;

public:
    explicit FunctionToSDFGPass(const PluginRegistry& plugin_registry);

    static bool available(analysis::AnalysisManager& AM) { return true; }

    llvm::PreservedAnalyses run(llvm::Module& Module, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM);
};

} // namespace passes
} // namespace docc
