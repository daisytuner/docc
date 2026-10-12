#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace axelera {

/// Emits an `A.grid` loop around one `A.Datapath` for a datapath loop whose body is a lane loop.
class DatapathDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& map_;

public:
    DatapathDispatcher(
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
