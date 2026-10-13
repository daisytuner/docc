#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/reduce.h>

namespace sdfg {
namespace axelera {

/**
 * Emits an AXELERA map as AxScript: a Core loop becomes one `A.AICore` region per core, a Stream loop an `A.grid`
 * loop around one `A.Datapath`, whose Lane loop is absorbed into the PWORD dtypes.
 */
class AxeleraMapDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& map_;

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

/**
 * Emits an AXELERA reduction as AxScript: one datapath accumulates the PWORDs in a DPU register and reduces its
 * lanes; on several cores each core stores its partial PWORD to L2 and the last core combines them after a barrier.
 * The surrounding accumulator initialization and result store are part of the datapath.
 */
class AxeleraReduceDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Reduce& reduce_;

public:
    AxeleraReduceDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Reduce& node,
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
