#pragma once

#include "c_style_base_code_generator.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"

namespace sdfg {
namespace codegen {

class CCodeGenerator : public CStyleBaseCodeGenerator {
private:
    CLanguageExtension language_extension_;

protected:
    void dispatch_header_includes(PrettyPrinter& out) override;

    void dispatch_header_structures(PrettyPrinter& out) override;

    void dispatch_globals() override;

    void dispatch_schedule() override;

    LanguageExtension& language_extension() override { return language_extension_; }

public:
    explicit CCodeGenerator(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan,
        std::shared_ptr<CodeSnippetFactory> library_snippet_factory = std::make_shared<CodeSnippetFactory>(),
        const std::string& externals_prefix = ""
    )
        : CStyleBaseCodeGenerator(
              sdfg,
              analysis_manager,
              instrumentation_plan,
              arg_capture_plan,
              std::move(library_snippet_factory),
              externals_prefix
          ),
          language_extension_(sdfg, externals_prefix) {}

    std::string function_definition() override;

    void emit_capture_context_init(std::ostream& ofs_source) const override;
};

} // namespace codegen
} // namespace sdfg
