#pragma once
#include "docc/util/docc_paths.h"

namespace docc::compile {

class SrcFileCompiler;

template<typename T>
class CodegenCompilerBuilderBase {
protected:
    std::shared_ptr<util::DefaultDoccPaths> docc_paths_;

public:
    virtual ~CodegenCompilerBuilderBase() = default;

    virtual T& set_from_paths(std::shared_ptr<util::DefaultDoccPaths> paths) {
        docc_paths_ = paths;
        return *dynamic_cast<T*>(this);
    }

    const util::DefaultDoccPaths& docc_paths() const { return *docc_paths_; }
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
    std::unordered_map<std::string, std::unique_ptr<SrcFileCompiler>> redirects_;
    std::vector<std::string> parent_link_options_;

public:
    SrcFileCompilerBuilder();
    ~SrcFileCompilerBuilder();

    SrcFileCompilerBuilder(SrcFileCompilerBuilder&&) noexcept;
    SrcFileCompilerBuilder& operator=(SrcFileCompilerBuilder&&) noexcept;

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
    SrcFileCompilerBuilder& redirect_snippet(const std::string& ext, SrcFileCompilerBuilder sub_builder);

    std::unique_ptr<SrcFileCompiler> build();

    bool remove_compile_option(const std::string& opt);
};

} // namespace docc::compile
