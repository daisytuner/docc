#include "docc/compile/src_file_compiler_builder.h"

#include "docc/compile/src_file_compiler.h"

namespace docc::compile {

SrcFileCompilerBuilder::SrcFileCompilerBuilder() {}

SrcFileCompilerBuilder::~SrcFileCompilerBuilder() = default;

SrcFileCompilerBuilder::SrcFileCompilerBuilder(SrcFileCompilerBuilder&&) noexcept = default;

SrcFileCompilerBuilder& SrcFileCompilerBuilder::operator=(SrcFileCompilerBuilder&&) noexcept = default;

SrcFileCompilerBuilder& SrcFileCompilerBuilder::set_src_extension(const std::string& ext) {
    main_src_ext_ = ext;
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::set_output_dir(const std::filesystem::path& out) {
    output_dir_ = out;
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::add_compile_option(const std::string& opt) {
    compile_options_.push_back(opt);
    return *this;
}

bool SrcFileCompilerBuilder::remove_compile_option(const std::string& opt) {
    auto it = std::find(compile_options_.begin(), compile_options_.end(), opt);
    if (it != compile_options_.end()) {
        compile_options_.erase(it);
        return true;
    }
    return false;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::set_link_immediately(bool link_imm) {
    link_immediately_ = link_imm;
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::add_link_option(const std::string& opt) {
    link_options_.push_back(opt);
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::add_common_option(const std::string& opt) {
    common_options_.push_back(opt);
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::add_include_path(const std::filesystem::path& path) {
    include_paths_.push_back(path);
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::add_library_path(const std::filesystem::path& path) {
    library_paths_.push_back(path);
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::set_compiler(const std::filesystem::path& compiler) {
    compiler_ = compiler;
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::set_linker(const std::filesystem::path& linker) {
    linker_ = linker;
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::set_from_paths(std::shared_ptr<util::DefaultDoccPaths> paths) {
    auto incs = paths->get_default_include_paths();
    include_paths_.insert(include_paths_.end(), incs.cbegin(), incs.cend());
    auto libs = paths->get_default_library_paths();
    library_paths_.insert(library_paths_.end(), libs.cbegin(), libs.cend());
    return CodegenCompilerBuilderBase::set_from_paths(paths);
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::inherit(const SrcFileCompilerBuilder& builder, bool compile_options) {
    output_dir_ = builder.output_dir_;
    compiler_ = builder.compiler_;
    docc_paths_ = builder.docc_paths_;

    if (compile_options) {
        main_src_ext_ = builder.main_src_ext_;
        include_paths_.insert(include_paths_.end(), builder.include_paths_.cbegin(), builder.include_paths_.cend());
        common_options_.insert(common_options_.end(), builder.common_options_.cbegin(), builder.common_options_.cend());
        compile_options_
            .insert(compile_options_.end(), builder.compile_options_.cbegin(), builder.compile_options_.cend());
    }

    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::set_bin_extension(const std::string& ext) {
    bin_ext_ = ext;
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::contribute_parent_link_options(const std::vector<std::string>& opts) {
    parent_link_options_.insert(parent_link_options_.end(), opts.cbegin(), opts.cend());
    return *this;
}

SrcFileCompilerBuilder& SrcFileCompilerBuilder::
    redirect_snippet(const std::string& ext, SrcFileCompilerBuilder sub_builder) {
    redirects_[ext] = sub_builder.build();
    return *this;
}

std::unique_ptr<SrcFileCompiler> SrcFileCompilerBuilder::build() {
    std::string compiler = this->compiler_.value();
    std::stringstream compiler_args;
    std::stringstream common_args;

    for (auto& inc : this->include_paths_) {
        compiler_args << "-I " << inc << " ";
    }
    for (auto& option : this->compile_options_) {
        compiler_args << option << " ";
    }
    for (auto& option : this->common_options_) {
        common_args << option << " ";
    }

    return std::make_unique<SrcFileCompiler>(
        output_dir_.value(),
        main_src_ext_.value(),
        "h",
        bin_ext_,
        compiler,
        this->linker_,
        common_args.str(),
        compiler_args.str(),
        library_paths_,
        link_options_,
        link_immediately_,
        std::move(this->redirects_),
        parent_link_options_
    );
}

} // namespace docc::compile
