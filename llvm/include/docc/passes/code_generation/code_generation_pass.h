#pragma once

#include <docc/docc_paths.h>
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <sdfg/codegen/code_snippet_factory.h>

#include "docc/analysis/analysis.h"
#include "docc/analysis/sdfg_registry.h"
#include "docc/passes/docc_pass.h"

namespace docc {
namespace passes {

class CodeGenerationPass : public llvm::PassInfoMixin<CodeGenerationPass> {
private:
    std::vector<std::string> sub_compile_opts_;

    void add_subcomp_opts_from_arg(std::vector<std::string> &out_opts) const;

    std::unique_ptr<llvm::Module> compile_to_ir_in_memory(
        llvm::LLVMContext &ctx,
        const std::filesystem::path &source_path,
        const std::vector<std::string> &compile_args,
        const std::vector<std::string> &add_compile_args,
        llvm::StringRef target_triple
    );

    bool write_library_snippets_to_files(
        std::filesystem::path build_path,
        std::unordered_set<std::string> lib_files,
        const std::unordered_map<std::string, sdfg::codegen::CodeSnippet> &snippets,
        std::unordered_map<std::string, std::vector<std::filesystem::path>> &files_for_post_processing
    );

    bool compile_additional_files(
        const std::filesystem::path &build_path,
        const std::vector<std::string> &compile_args,
        std::set<std::string> &link_2nd_args,
        std::unordered_map<std::string, std::vector<std::filesystem::path>> files_for_post_processing,
        const utils::DoccPaths &docc_paths
    );

    bool generate_code(
        llvm::Module &Mod,
        sdfg::StructuredSDFG &sdfg,
        const analysis::Attributes &attributes,
        llvm::dwarf::SourceLanguage &language,
        const std::vector<std::string> &mod_wide_compile_flags,
        std::set<std::string> &link_2nd_args,
        utils::DoccPaths docc_paths
    );

    bool link_into_existing_module(
        llvm::Module &Mod,
        const std::vector<std::string> &compile_args,
        const std::vector<std::string> &add_compiler_args,
        const std::filesystem::path &source_path,
        const std::string &steal_from
    );

    std::list<std::unique_ptr<sdfg::StructuredSDFG>>
    split_sdfg(sdfg::StructuredSDFG &sdfg, const analysis::Attributes &attributes);

public:
    CodeGenerationPass();

    static bool available(analysis::AnalysisManager &AM) { return analysis::SDFGRegistry::is_link_time(AM); }

    llvm::PreservedAnalyses run(llvm::Module &Module, llvm::ModuleAnalysisManager &MAM, analysis::AnalysisManager &AM);

    bool compile_to_object_file(
        const std::filesystem::path &source_path,
        const std::filesystem::path &object_file,
        const std::vector<std::string> &compile_args
    );
};

} // namespace passes
} // namespace docc
