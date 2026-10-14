#pragma once

#include <set>

#include <sdfg/codegen/code_generator.h>
#include <sdfg/structured_control_flow/structured_loop.h>

#include "docc/target/axelera/axscript_language_extension.h"

namespace sdfg {
namespace axelera {

/**
 * Generates an AxScript module: an `@A.ir_module` class named after the SDFG whose `main` takes the read buffers
 * and returns the written ones through `A.output`. Only scheduled datapath loops may appear at the top level; the
 * arrays and reduction results they pass each other live in DDR buffers of the module.
 */
class AxScriptCodeGenerator : public codegen::CodeGenerator {
private:
    struct Buffer {
        std::string name;
        int64_t elements;
        bool is_output;
        /// Elements per row of a matrix buffer; 0 for a flat one.
        int64_t row_length = 0;
        types::PrimitiveType element = types::PrimitiveType::Float;
        /// Allocated by the module rather than passed in or returned.
        bool transient = false;

        int64_t pwords() const;
        /// The PWORD shape AxScript declares, e.g. "(8,)" or "(128, 2)".
        std::string shape() const;
    };

    AxScriptLanguageExtension language_extension_;
    /// Datapath loops, AI-core programs and transposing copies, in program order.
    std::vector<structured_control_flow::ControlFlowNode*> datapaths_;
    std::vector<Buffer> buffers_;
    std::set<std::string> reduction_results_;
    int64_t num_cores_ = 1;
    bool uses_operators_ = false;

    /// Checks that the function body consists of datapath loops only, with reductions' init and store blocks.
    void collect_datapaths();

    void collect_buffers();

    /// A transposing tensor copy as a DWPU vector transpose, split over the AI cores.
    void emit_transpose(codegen::PrettyPrinter& out, structured_control_flow::Block& block) const;

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
