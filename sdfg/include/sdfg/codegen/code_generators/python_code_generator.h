#pragma once

#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/language_extensions/python_language_extension.h"

namespace sdfg {
namespace codegen {

/**
 * Generates a self-contained Python module whose function of the SDFG's name takes NumPy arrays
 * (flat views for pointers) and scalars. Supports sequential, vectorized and serializable schedules only; library
 * nodes without a Python dispatcher must be expanded beforehand.
 */
class PythonCodeGenerator : public CodeGenerator {
private:
    PythonLanguageExtension language_extension_;

    void validate(structured_control_flow::ControlFlowNode& node) const;

    void dispatch_locals();

public:
    PythonCodeGenerator(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan,
        std::shared_ptr<CodeSnippetFactory> library_snippet_factory = std::make_shared<CodeSnippetFactory>(),
        const std::string& externals_prefix = ""
    );

    bool generate() override;

    std::string function_definition() override;

    /// Python has no headers; nothing is emitted.
    bool emit_header(PrettyPrinter& out) override;

    bool emit_main_source(std::ostream& out, const std::filesystem::path& header_path) override;

    /// Writes the module to @p source_path; @p header_path is ignored.
    bool as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) override;

    void append_function_source(std::ostream& ofs_source) override;
};

} // namespace codegen
} // namespace sdfg
