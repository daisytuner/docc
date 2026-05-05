#pragma once

#include "sdfg/codegen/code_generator.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"

namespace sdfg {
namespace codegen {

class CUDACodeGenerator : public CodeGenerator {
private:
    CUDALanguageExtension language_extension_;

protected:
    void dispatch_includes();
    void dispatch_header_includes(PrettyPrinter& out);
    void dispatch_structures();
    void dispatch_header_structures(PrettyPrinter& out);

    void dispatch_globals();

    void dispatch_schedule();

public:
    CUDACodeGenerator(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan,
        std::shared_ptr<CodeSnippetFactory> library_snippet_factory = std::make_shared<CodeSnippetFactory>()
    );

    bool generate() override;

    std::string function_definition() override;

    bool emit_header(PrettyPrinter& out) override;
    bool emit_main_source(std::ostream& out, const std::filesystem::path& header_path) override;

    bool as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) override;

    void append_function_source(std::ostream& ofs_source) override;
};

} // namespace codegen
} // namespace sdfg
