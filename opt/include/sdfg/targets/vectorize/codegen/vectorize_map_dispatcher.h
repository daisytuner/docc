#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/codegen/dispatchers/node_dispatcher.h>
#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/symbolic/symbolic.h>

namespace sdfg {
namespace vectorize {

class VectorizeMapDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& node_;

public:
    VectorizeMapDispatcher(
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

    codegen::InstrumentationInfo instrumentation_info() const override;
};

} // namespace vectorize
} // namespace sdfg
