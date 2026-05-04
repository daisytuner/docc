#pragma once

#include <filesystem>
#include <string>
#include <utility>

#include "code_snippet_factory.h"
#include "sdfg/analysis/mem_access_range_analysis.h"
#include "sdfg/codegen/instrumentation/arg_capture_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace codegen {

/**
 * @brief Base class for code generators
 *
 * This class provides the basic structure for code generators.
 * It contains the streams for the includes, classes, globals, library functions and main code.
 */
class CodeGenerator {
protected:
    /// @brief Reference to the schedule
    StructuredSDFG& sdfg_;

    /// @brief Analysis manager for the SDFG
    analysis::AnalysisManager& analysis_manager_;

    /// @brief Instrumentation strategy
    InstrumentationPlan& instrumentation_plan_;
    ArgCapturePlan& arg_capture_plan_;

    /// @brief Stream for includes
    PrettyPrinter includes_stream_;

    /// @brief Stream for classes
    PrettyPrinter classes_stream_;

    /// @brief Stream for global variables and functions
    PrettyPrinter globals_stream_;

    /// @brief Stream for library functions
    std::shared_ptr<CodeSnippetFactory> library_snippet_factory_;

    /// @brief Main stream
    PrettyPrinter main_stream_;

    /// @brief Prefix to use for external functions
    std::string externals_prefix_;

public:
    CodeGenerator(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan,
        std::shared_ptr<CodeSnippetFactory> library_snippet_factory = std::make_shared<CodeSnippetFactory>(),
        const std::string& externals_prefix = ""
    )
        : sdfg_(sdfg), analysis_manager_(analysis_manager), instrumentation_plan_(instrumentation_plan),
          arg_capture_plan_(arg_capture_plan), library_snippet_factory_(std::move(library_snippet_factory)),
          externals_prefix_(externals_prefix) {};


    virtual ~CodeGenerator() = default;

    [[nodiscard]] const StructuredSDFG& sdfg() const { return this->sdfg_; };

    /**
     * @brief Generate the code
     *
     * @return true if the code was generated successfully
     */
    virtual bool generate() = 0;

    /// @brief Generate a function definition for the SDFG
    virtual std::string function_definition() = 0;

    /// @brief write the contents of the main header into out
    virtual bool emit_header(PrettyPrinter& out) = 0;

    /// @brief write the contents of the main source file into out. Can also generate additional source files (snippets)
    /// that also depend on the header
    virtual bool emit_main_source(std::ostream& out, const std::filesystem::path& header_path) = 0;

    /// @brief Generate the SDFG's code into source files
    virtual bool as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) = 0;

    /// @brief Generate only the function source code and append it to the source file. @ref as_source generates this
    /// into `source_path` after includes, globals and structs
    virtual void append_function_source(std::ostream& ofs_source) = 0;

    /// @brief Get the includes
    const PrettyPrinter& includes() const { return this->includes_stream_; };

    /// @brief Get the classes
    const PrettyPrinter& classes() const { return this->classes_stream_; };

    /// @brief Get the globals
    const PrettyPrinter& globals() const { return this->globals_stream_; };

    /// @brief all created library snippets
    const std::unordered_map<std::string, CodeSnippet>& library_snippets() const {
        return this->library_snippet_factory_->snippets();
    };

    /// @brief Get the main stream
    const PrettyPrinter& main() const { return this->main_stream_; };
};

} // namespace codegen
} // namespace sdfg
