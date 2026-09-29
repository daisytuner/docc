#pragma once

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace codegen {

/**
 * Dispatches to the actual NodeDispatcher based on scheduleType and the MapDispatcherRegistry
 */
class SchedTypeMapDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Map& node_;

public:
    SchedTypeMapDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override {
        throw std::runtime_error("MapDispatcher::dispatch_node not implemented");
    }

    void dispatch(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;

    InstrumentationInfo instrumentation_info() const override;
};

/**
 * @deprecated Use the new nyme [SchedTypeMapDispatcher]. The old name suggests it is perhaps a base class for other
 * dispatchers, which it is not
 */
typedef SchedTypeMapDispatcher MapDispatcher;

class SequentialMapDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Map& node_;

public:
    SequentialMapDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;

    InstrumentationInfo instrumentation_info() const override;
};

} // namespace codegen
} // namespace sdfg
