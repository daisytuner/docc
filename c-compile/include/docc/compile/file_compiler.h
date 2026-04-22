#pragma once
#include "docc/compile/codegen_build_pool.h"
#include "docc/util/docc_paths.h"

namespace docc::compile {

class SrcFileCompiler;

class FileCompileState : public CompileState {
    friend class SrcFileCompiler;
    SrcFileCompiler& compiler_;
    std::filesystem::path src_path_;
    std::filesystem::path out_path_;
    bool link_immediately_;
    std::function<void(std::ostream&)> generator_;

    bool src_done_ = false;
    bool obj_done_ = false;

public:
    FileCompileState(
        SrcFileCompiler& compiler,
        const std::filesystem::path& src_path,
        const std::filesystem::path& obj_path,
        bool link_immediately,
        std::function<void(std::ostream&)>& generator
    );

    [[nodiscard]] CodegenCompiler& creator() const override;

    bool codegen() override;
    bool compile() override;

    const std::filesystem::path& out_path() const;

    bool has_obj_to_link() const;
};

class SrcFileCompiler : public CodegenCompiler {
    friend class FileCompileState;

    std::mutex mutex_;
    std::filesystem::path output_dir_;
    std::string compiler_;
    std::optional<std::string> linker_;
    std::string common_args_;
    std::string compile_args_;
    std::vector<std::filesystem::path> library_paths_;
    std::vector<std::string> link_options_;
    std::string main_src_ext_;
    std::string main_header_ext_;
    std::string bin_ext_;
    bool link_immediately_;

    inline static constexpr auto COMPILE_ONLY_FLAG = "-c";
    inline static constexpr auto OUTPUT_FILE_ARG = "-o";

public:
    SrcFileCompiler(
        const std::filesystem::path& output_dir,
        const std::string& main_src_ext,
        const std::string& main_header_ext,
        const std::string& bin_ext,
        const std::string& compiler,
        const std::optional<std::string>& linker,
        const std::string& common_args,
        const std::string& compile_args,
        const std::vector<std::filesystem::path>& library_paths,
        const std::vector<std::string>& link_options,
        bool link_immediately,
        std::unordered_map<std::string, std::unique_ptr<CodegenCompiler>>&& redirects
    );

    std::unique_ptr<CompileState> do_create_compile(
        const sdfg::StructuredSDFG& sdfg,
        const sdfg::codegen::CodeSnippet* snippet,
        std::function<void(std::ostream&)> generator
    ) override;

    std::filesystem::path
    process(sdfg::codegen::CodeGenerator& generator, CompileExecutor& executor, const std::string& output_file_name);
    std::filesystem::path process(sdfg::codegen::CodeGenerator& generator, CompileExecutor& executor) {
        return process(generator, executor, generator.sdfg().name());
    }

    std::shared_ptr<sdfg::codegen::CodeSnippetFactory> create_snippet_factory(const sdfg::StructuredSDFG& sdfg) const;

protected:
    std::filesystem::path generate_header_path(const sdfg::StructuredSDFG& sdfg) const;
    bool run_compiler(const std::filesystem::path& src, const std::filesystem::path& obj) const;

    void add_link_args(std::stringstream& cmd) const;

    bool run_link(const sdfg::StructuredSDFG& sdfg, CompileExecutor& executor, const std::filesystem::path& bin_file)
        const;
    bool run_compile_and_link_single(const std::filesystem::path& src, const std::filesystem::path& bin) const;
    std::filesystem::path emit_header(const sdfg::StructuredSDFG& sdfg, sdfg::codegen::CodeGenerator& generator);
    void for_each_file_snippet(
        sdfg::codegen::CodeGenerator& generator, std::function<void(const sdfg::codegen::CodeSnippet&)> callback
    );
};

class SrcFileCompilerBuilder : public CodegenCompilerBuilderBase<SrcFileCompilerBuilder> {
    std::optional<std::filesystem::path> output_dir_;
    std::optional<std::filesystem::path> compiler_;
    std::optional<std::filesystem::path> linker_;
    std::optional<std::string> main_src_ext_;
    std::string bin_ext_ = "elf";
    std::vector<std::string> common_options_;
    std::vector<std::string> compile_options_;
    std::vector<std::string> link_options_;
    std::vector<std::filesystem::path> include_paths_;
    std::vector<std::filesystem::path> library_paths_;
    bool link_immediately_ = false;
    std::vector<std::string> parent_link_options_;

public:
    SrcFileCompilerBuilder();

    SrcFileCompilerBuilder& set_src_extension(const std::string& ext);
    SrcFileCompilerBuilder& set_output_dir(const std::filesystem::path& output_dir);
    SrcFileCompilerBuilder& add_compile_option(const std::string& opt);
    SrcFileCompilerBuilder& add_link_option(const std::string& opt);
    SrcFileCompilerBuilder& add_common_option(const std::string& opt);
    SrcFileCompilerBuilder& add_include_path(const std::filesystem::path& path);
    SrcFileCompilerBuilder& add_library_path(const std::filesystem::path& path);
    SrcFileCompilerBuilder& set_compiler(const std::filesystem::path& compiler);
    SrcFileCompilerBuilder& set_linker(const std::filesystem::path& linker);
    SrcFileCompilerBuilder& set_from_paths(std::shared_ptr<util::DefaultDoccPaths> paths) override;
    SrcFileCompilerBuilder& set_bin_extension(const std::string& ext);
    SrcFileCompilerBuilder& inherit(const SrcFileCompilerBuilder& builder, bool compile_options = false);
    /**
     * There is only 1 source and no need to compile an object file and then link all object files.
     * Fuse compile and link options. Expects the [compiler-executable] to be a gcc-like driver that can also handle
     * linking
     */
    SrcFileCompilerBuilder& set_link_immediately(bool link_imm);
    SrcFileCompilerBuilder& contribute_parent_link_options(const std::vector<std::string>& opts);

    std::unique_ptr<SrcFileCompiler> build();

    bool remove_compile_option(const std::string& opt);
};

} // namespace docc::compile
