#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace axelera {

class AxScriptLanguageExtension;

/**
 * Emits AXELERA loops as AxScript: a Core loop becomes one `A.AICore` region per core, a Stream loop an `A.grid`
 * loop around one `A.Datapath`, whose Lane loop is absorbed into the PWORD dtypes.
 */
class AxeleraMapDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& map_;

    void dispatch_cores(AxScriptLanguageExtension& lang, codegen::PrettyPrinter& stream);

    /// @param first_element Element offset of the first PWORD, with any enclosing core index substituted.
    void dispatch_stream(
        AxScriptLanguageExtension& lang,
        codegen::PrettyPrinter& stream,
        structured_control_flow::Map& stream_loop,
        const symbolic::Expression& first_element
    );

public:
    AxeleraMapDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        codegen::PrettyPrinter& main_stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};

} // namespace axelera
} // namespace sdfg
