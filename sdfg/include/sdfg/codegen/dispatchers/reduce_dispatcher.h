#pragma once

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/structured_control_flow/reduce.h"

namespace sdfg {
namespace codegen {

/**
 * @brief Dispatches to the actual NodeDispatcher based on the Reduce's schedule
 *        type and the ReduceDispatcherRegistry
 *
 * Mirrors @ref SchedTypeMapDispatcher: this is the dispatcher registered for
 * `typeid(structured_control_flow::Reduce)` in the NodeDispatcherRegistry. It
 * looks up the concrete, schedule-specific dispatcher (sequential, OpenMP, ...)
 * in the ReduceDispatcherRegistry and delegates to it.
 */
class SchedTypeReduceDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Reduce& node_;

public:
    SchedTypeReduceDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Reduce& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override {
        throw std::runtime_error("ReduceDispatcher::dispatch_node not implemented");
    }

    void dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory)
        override;

    InstrumentationInfo instrumentation_info() const override;
};

/**
 * @deprecated Use the new name [SchedTypeReduceDispatcher]. The old name suggests it is perhaps a base class for other
 * dispatchers, which it is not
 */
typedef SchedTypeReduceDispatcher ReduceDispatcher;

/**
 * @brief Default (sequential) code generator for @ref structured_control_flow::Reduce
 *
 * A Reduce loop is semantically a sequential loop whose body combines values
 * into an accumulator. Lowering it like a plain sequential for-loop is always
 * correct because the combine and accumulator remain in the body.
 */
class SequentialReduceDispatcher : public NodeDispatcher {
private:
    structured_control_flow::Reduce& node_;

public:
    SequentialReduceDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Reduce& node,
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
