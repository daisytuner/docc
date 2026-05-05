#include "docc/compile/src_file_compiler.h"

#include "docc/util/docc_paths.h"

namespace docc::compile {

FileCompileState::FileCompileState(
    SrcFileCompiler& compiler,
    const sdfg::codegen::CodeSnippet* snippet,
    const std::filesystem::path& src_path,
    const std::filesystem::path& out_path,
    bool link_immediately,
    std::function<void(std::ostream&)>& generator
)
    : CompileState(), compiler_(compiler), snippet_(snippet), src_path_(src_path), out_path_(out_path),
      link_immediately_(link_immediately), generator_(generator), src_done_(generator == nullptr) {}

CodegenCompiler& FileCompileState::creator() const { return compiler_; }

bool FileCompileState::codegen() {
    if (generator_) {
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        std::ofstream outfile(src_path_, std::ofstream::out | std::ofstream::trunc);

        generator_(outfile);

        outfile.close();
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        src_done_ = true;
        gen_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    }
    return true;
}

bool FileCompileState::compile() {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    auto success = link_immediately_ ? compiler_.run_compile_and_link_single(src_path_, out_path_)
                                     : compiler_.run_compiler(src_path_, out_path_);
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    if (success) {
        obj_done_ = true;
    }
    build_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    return success;
}

const std::filesystem::path& FileCompileState::out_path() const { return out_path_; }

bool FileCompileState::has_obj_to_link() const { return !link_immediately_; }

void FileCompileState::record_stats(const sdfg::StructuredSDFG& sdfg, sdfg::passes::CodegenStatistics& stats) {
    auto name = snippet_ ? sdfg.name() + ":" + snippet_->name() + "." + snippet_->extension() : sdfg.name();
    stats.add_codegen(name + "@gen", gen_time_.count());

    auto build_name = name + (link_immediately_ ? "@build" : "@compile");
    stats.add_codegen(name, build_time_.count());
}

SrcFileCompiler::SrcFileCompiler(
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
)
    : output_dir_(output_dir), main_src_ext_(main_src_ext), main_header_ext_(main_header_ext), bin_ext_(bin_ext),
      compiler_(compiler), linker_(linker), common_args_(common_args), compile_args_(compile_args),
      library_paths_(library_paths), link_options_(link_options), link_immediately_(link_immediately),
      redirects_(std::move(redirects)), parent_link_opts_(parent_link_options) {
    auto& codegen_statistics = sdfg::passes::CodegenStatistics::instance();
    if (codegen_statistics.enabled()) {
        stats_ = &codegen_statistics;
    }
}

std::filesystem::path SrcFileCompiler::
    emit_header(const sdfg::StructuredSDFG& sdfg, sdfg::codegen::CodeGenerator& generator) {
    auto p = generate_header_path(sdfg);
    std::ofstream header_file(p, std::ofstream::out | std::ofstream::trunc);
    sdfg::codegen::PrettyPrinter os(header_file);
    generator.emit_header(os);

    header_file.close();

    return p;
}

std::unique_ptr<CompileState> SrcFileCompiler::create_compile(
    const sdfg::StructuredSDFG& sdfg,
    const sdfg::codegen::CodeSnippet* snippet,
    std::function<void(std::ostream&)> generator
) {
    if (snippet) {
        auto& ext = snippet->extension();
        auto it = redirects_.find(ext);
        if (it != redirects_.end()) {
            return it->second->do_create_compile(sdfg, snippet, generator);
        }
    }

    return do_create_compile(sdfg, snippet, generator);
}

std::unique_ptr<CompileState> SrcFileCompiler::do_create_compile(
    const sdfg::StructuredSDFG& sdfg,
    const sdfg::codegen::CodeSnippet* snippet,
    std::function<void(std::ostream&)> generator
) {
    const std::string* name;
    const std::string* ext;
    if (snippet) {
        name = &snippet->name();
        ext = &snippet->extension();
    } else {
        name = &sdfg.name();
        ext = &main_src_ext_;
    }
    const std::string& out_ext = (link_immediately_ ? bin_ext_ : "o");

    auto state = std::make_unique<FileCompileState>(
        *this,
        snippet,
        output_dir_ / (*name + "." + *ext),
        output_dir_ / (*name + "." + out_ext),
        link_immediately_,
        generator
    );

    return std::move(state);
}

std::filesystem::path SrcFileCompiler::generate_header_path(const sdfg::StructuredSDFG& sdfg) const {
    auto header_path = output_dir_ / (sdfg.name() + "." + main_header_ext_);
    return header_path;
}

std::shared_ptr<sdfg::codegen::CodeSnippetFactory> SrcFileCompiler::create_snippet_factory(const sdfg::StructuredSDFG&
                                                                                               sdfg) const {
    auto header_path = generate_header_path(sdfg);
    std::pair<std::filesystem::path, std::filesystem::path> lib_config = std::make_pair(output_dir_, header_path);
    return std::make_shared<sdfg::codegen::CodeSnippetFactory>(&lib_config);
}

/**
 * this is to hide library snippet handling. In the future we want this to support asynchronously delivering finished
 * snippets, so we can compile them as soon as possible. But right now, they are all available after the main code
 * generation completed, so we do not need to do any of that.
 */
void SrcFileCompiler::for_each_file_snippet(
    sdfg::codegen::CodeGenerator& generator,
    CompileExecutor& executor,
    std::function<void(const sdfg::codegen::CodeSnippet&)> callback
) {
    // for now, the snippets do not arrive asynchronously, so we need to await the main-dispatch here, before the
    // snippets will exist
    executor.await_compiles_finished();

    for (auto& [id, snippet] : generator.library_snippets()) {
        if (snippet.is_as_file()) {
            callback(snippet);
        }
    }
}

std::filesystem::path SrcFileCompiler::
    process(sdfg::codegen::CodeGenerator& generator, CompileExecutor& executor, const std::string& output_file_name) {
    auto& sdfg = generator.sdfg();
    auto header_path = emit_header(sdfg, generator);

    auto mainCompile = create_compile(sdfg, nullptr, [&](std::ostream& out) {
        generator.emit_main_source(out, header_path);
    });

    executor.add_compile_state(std::move(mainCompile));

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for_each_file_snippet(generator, executor, [&](const sdfg::codegen::CodeSnippet& snippet) {
        auto compile_state = create_compile(sdfg, &snippet, [&](std::ostream& out) { out << snippet.stream().str(); });

        executor.add_compile_state(std::move(compile_state));
    });

    executor.await_compiles_finished();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    if (stats_) {
        std::string name = "snippet_build_total";
        if (executor.is_parallel()) {
            name += "@parallel";
        }
        stats_->add_codegen(name, std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }

    std::filesystem::path bin_file = output_dir_ / (output_file_name + "." + bin_ext_);

    run_link(sdfg, executor, bin_file);

    return bin_file;
}

bool SrcFileCompiler::run_compiler(const std::filesystem::path& src, const std::filesystem::path& obj) const {
    std::stringstream cmd;

    cmd << compiler_ << " " << COMPILE_ONLY_FLAG << " " << common_args_ << " " << compile_args_ << " " << src << " "
        << OUTPUT_FILE_ARG << " " << obj;

    DEBUG_PRINTLN("Compile: " << cmd.str());
    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        throw std::runtime_error("Compilation failed: " + cmd.str());
    }
    return true;
}

bool SrcFileCompiler::run_compile_and_link_single(const std::filesystem::path& src, const std::filesystem::path& bin)
    const {
    std::stringstream cmd;

    cmd << compiler_ << " " << common_args_ << " " << compile_args_ << " ";
    cmd << src << " -o " << bin << " ";
    add_link_args(cmd);

    DEBUG_PRINTLN("Build: " << cmd.str());
    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        throw std::runtime_error("Build failed: " + cmd.str());
    }

    return true;
}

void SrcFileCompiler::add_link_args(std::stringstream& cmd) const {
    for (auto& ld : this->library_paths_) {
        cmd << "-L " << ld << " ";
    }
    for (auto& option : this->link_options_) {
        cmd << option << " ";
    }
}

bool SrcFileCompiler::
    run_link(const sdfg::StructuredSDFG& sdfg, CompileExecutor& executor, const std::filesystem::path& bin_file) const {
    std::stringstream cmd;

    cmd << linker_.value_or(compiler_) << " ";

    cmd << common_args_ << " ";

    std::unordered_map<LinkOptContributor*, int> sub_counts;
    std::vector<std::string> sub_opts;

    executor.for_each_src([&](CompileState& state) {
        auto& file_state = dynamic_cast<FileCompileState&>(state);
        if (file_state.has_obj_to_link()) {
            if (stats_) {
                file_state.record_stats(sdfg, *stats_);
            }
            assert(file_state.obj_done_);
            cmd << file_state.out_path_ << " ";

            auto* possible_contrib = dynamic_cast<LinkOptContributor*>(&file_state.creator());
            if (possible_contrib) {
                auto& count = sub_counts[possible_contrib];
                if (count == 0) {
                    auto* contrib_opts = possible_contrib->get_contributed_link_options();
                    if (contrib_opts) {
                        sub_opts.insert(sub_opts.end(), contrib_opts->cbegin(), contrib_opts->cend());
                    }
                }
                ++count;
            }
        }
    });

    add_link_args(cmd);
    for (auto& opt : sub_opts) {
        cmd << opt << " ";
    }

    cmd << "-o " << bin_file;

    DEBUG_PRINTLN("Link: " << cmd.str());
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    int ret = std::system(cmd.str().c_str());
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    if (ret != 0) {
        throw std::runtime_error("Link failed: " + cmd.str());
    }
    if (stats_) {
        stats_->add_codegen(
            sdfg.name() + "@link", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        );
    }

    return true;
}

const std::vector<std::string>* SrcFileCompiler::get_contributed_link_options() const {
    if (parent_link_opts_.empty()) {
        return nullptr;
    } else {
        return &parent_link_opts_;
    }
}

} // namespace docc::compile
