#pragma once

#include <sdfg/codegen/code_generator.h>
#include <sdfg/structured_control_flow/map.h>

#include "docc/target/axelera/axscript_language_extension.h"

namespace sdfg {
namespace axelera {

/**
 * Generates an AxScript module: an `@A.ir_module` class named after the SDFG whose `main` takes the read buffers
 * and returns the written ones through `A.output`. Only scheduled datapath loops may appear at the top level.
 */
class AxScriptCodeGenerator : public codegen::CodeGenerator {
private:
    struct Buffer {
        std::string name;
        int64_t pwords;
        bool is_output;
    };

    AxScriptLanguageExtension language_extension_;
    std::vector<structured_control_flow::Map*> datapaths_;
    std::vector<Buffer> buffers_;
    int64_t num_cores_ = 1;

    /// Checks that the function body consists of datapath loops only.
    void collect_datapaths();

    void collect_buffers();

    std::string buffer_type(const Buffer& buffer) const;

public:
    AxScriptCodeGenerator(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan,
        std::shared_ptr<codegen::CodeSnippetFactory> library_snippet_factory =
            std::make_shared<codegen::CodeSnippetFactory>(),
        const std::string& externals_prefix = ""
    );

    bool generate() override;

    std::string function_definition() override;

    /// AxScript has no headers; nothing is emitted.
    bool emit_header(codegen::PrettyPrinter& out) override;

    bool emit_main_source(std::ostream& out, const std::filesystem::path& header_path) override;

    /// Writes the module to @p source_path; @p header_path is ignored.
    bool as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) override;

    void append_function_source(std::ostream& ofs_source) override;
};

} // namespace axelera
} // namespace sdfg
