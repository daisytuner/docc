#pragma once
#include <docc/compile/codegen_compiler.h>

#include "docc/compile/codegen_build_pool.h"
#include "docc/util/docc_paths.h"

namespace docc::compile {

class SrcFileCompiler;

class FileCompileState : public CompileState {
    friend class SrcFileCompiler;
    SrcFileCompiler& compiler_;
    const sdfg::codegen::CodeSnippet* snippet_;
    std::filesystem::path src_path_;
    std::filesystem::path out_path_;
    bool link_immediately_;
    std::function<void(std::ostream&)> generator_;

    bool src_done_ = false;
    bool obj_done_ = false;

    std::chrono::duration<std::chrono::milliseconds::rep, std::chrono::milliseconds::period> gen_time_;
    std::chrono::duration<std::chrono::milliseconds::rep, std::chrono::milliseconds::period> build_time_;

public:
    FileCompileState(
        SrcFileCompiler& compiler,
        const sdfg::codegen::CodeSnippet* snippet,
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

    void record_stats(const sdfg::StructuredSDFG& sdfg, sdfg::passes::CodegenStatistics& stats);
};

class LinkOptContributor {
public:
    virtual const std::vector<std::string>* get_contributed_link_options() const = 0;
};

class SrcFileCompiler : public CodegenCompiler, public LinkOptContributor {
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
    std::unordered_map<std::string, std::unique_ptr<SrcFileCompiler>> redirects_;
    std::vector<std::string> parent_link_opts_;
    sdfg::passes::CodegenStatistics* stats_ = nullptr;

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
        std::unordered_map<std::string, std::unique_ptr<SrcFileCompiler>>&& redirects,
        const std::vector<std::string>& parent_link_options
    );

    std::unique_ptr<CompileState> create_compile(
        const sdfg::StructuredSDFG& sdfg,
        const sdfg::codegen::CodeSnippet* snippet,
        std::function<void(std::ostream&)> generator
    ) override;

    std::unique_ptr<CompileState> do_create_compile(
        const sdfg::StructuredSDFG& sdfg,
        const sdfg::codegen::CodeSnippet* snippet,
        std::function<void(std::ostream&)> generator
    );

    std::filesystem::path
    process(sdfg::codegen::CodeGenerator& generator, CompileExecutor& executor, const std::string& output_file_name);
    std::filesystem::path process(sdfg::codegen::CodeGenerator& generator, CompileExecutor& executor) {
        return process(generator, executor, generator.sdfg().name());
    }

    std::shared_ptr<sdfg::codegen::CodeSnippetFactory> create_snippet_factory(const sdfg::StructuredSDFG& sdfg) const;

    const std::vector<std::string>* get_contributed_link_options() const override;

protected:
    std::filesystem::path generate_header_path(const sdfg::StructuredSDFG& sdfg) const;
    bool run_compiler(const std::filesystem::path& src, const std::filesystem::path& obj) const;

    void add_link_args(std::stringstream& cmd) const;

    bool run_link(const sdfg::StructuredSDFG& sdfg, CompileExecutor& executor, const std::filesystem::path& bin_file)
        const;
    bool run_compile_and_link_single(const std::filesystem::path& src, const std::filesystem::path& bin) const;
    std::filesystem::path emit_header(const sdfg::StructuredSDFG& sdfg, sdfg::codegen::CodeGenerator& generator);
    void for_each_file_snippet(
        sdfg::codegen::CodeGenerator& generator,
        CompileExecutor& executor,
        std::function<void(const sdfg::codegen::CodeSnippet&)> callback
    );
};

} // namespace docc::compile
