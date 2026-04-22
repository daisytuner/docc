
#include "sdfg/codegen/code_generators/c_style_base_code_generator.h"

#include "sdfg/helpers/helpers.h"

namespace sdfg::codegen {

CStyleBaseCodeGenerator::CStyleBaseCodeGenerator(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan,
    std::shared_ptr<CodeSnippetFactory> library_snippet_factory,
    const std::string& externals_prefix
)
    : CodeGenerator(
          sdfg,
          analysis_manager,
          instrumentation_plan,
          arg_capture_plan,
          std::move(library_snippet_factory),
          externals_prefix
      ) {
    if (sdfg.type() != FunctionType_CPU) {
        throw std::runtime_error("CStyleBaseCodeGenerator can only be used for CPU SDFGs");
    }
};

bool CStyleBaseCodeGenerator::generate() {
    this->dispatch_includes();
    this->dispatch_structures();
    this->dispatch_globals();
    this->dispatch_schedule();
    return true;
}

bool CStyleBaseCodeGenerator::emit_header(PrettyPrinter& out) {
    out << "#pragma once" << std::endl;

    dispatch_header(out);
    return true;
}

void CStyleBaseCodeGenerator::dispatch_header(PrettyPrinter& out) {
    dispatch_header_includes(out);
    dispatch_header_structures(out);
}

void CStyleBaseCodeGenerator::dispatch_includes() { dispatch_header_includes(this->includes_stream_); }

void CStyleBaseCodeGenerator::dispatch_structures() { dispatch_header_structures(this->classes_stream_); }

bool CStyleBaseCodeGenerator::emit_main_source(std::ostream& out, const std::filesystem::path& header_path) {
    out << "#include \"" << header_path.filename().string() << "\"" << std::endl;
    dispatch_globals();
    dispatch_schedule();

    for (auto& snippet : library_snippet_factory_->globals_snippets()) {
        out << snippet << std::endl;
    }

    out << this->globals_stream_.str() << std::endl;

    append_function_source(out);
    return true;
}

bool CStyleBaseCodeGenerator::as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) {
    std::ofstream ofs_header(header_path, std::ofstream::out | std::ofstream::trunc);
    if (!ofs_header.is_open()) {
        return false;
    }

    std::ofstream ofs_source(source_path, std::ofstream::out | std::ofstream::trunc);
    if (!ofs_source.is_open()) {
        return false;
    }

    ofs_header << "#pragma once" << std::endl;
    ofs_header << this->includes_stream_.str() << std::endl;
    ofs_header << this->classes_stream_.str() << std::endl;
    ofs_header.close();

    ofs_source << "#include \"" << header_path.filename().string() << "\"" << std::endl;
    for (auto& snippet : library_snippet_factory_->globals_snippets()) {
        ofs_source << snippet << std::endl;
    }
    ofs_source << this->globals_stream_.str() << std::endl;

    append_function_source(ofs_source);

    ofs_source.close();

    return true;
}

void CStyleBaseCodeGenerator::append_function_source(std::ostream& ofs_source) {
    std::unique_ptr<std::vector<CaptureVarPlan>> capturePlan;
    if (!arg_capture_plan_.is_empty()) {
        this->emit_capture_context_init(ofs_source);
    }

    ofs_source << this->function_definition() << std::endl;
    ofs_source << "{" << std::endl;

    auto init_once = library_snippet_factory_->find(CODE_SNIPPET_INIT_ONCE);
    if (init_once != library_snippet_factory_->snippets().end()) {
        ofs_source << init_once->second.stream().str() << std::endl;
    }

    for (auto& snippet : library_snippet_factory_->setup_snippets()) {
        ofs_source << snippet << std::endl;
    }

    ofs_source << this->main_stream_.str() << std::endl;

    for (auto& snippet : library_snippet_factory_->teardown_snippets()) {
        ofs_source << snippet << std::endl;
    }

    auto deinit_once = library_snippet_factory_->find(CODE_SNIPPET_DEINIT_ONCE);
    if (deinit_once != library_snippet_factory_->snippets().end()) {
        ofs_source << deinit_once->second.stream().str() << std::endl;
    }

    ofs_source << "}" << std::endl;
}

} // namespace sdfg::codegen
