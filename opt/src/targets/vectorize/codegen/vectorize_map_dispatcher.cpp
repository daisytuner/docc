#include "sdfg/targets/vectorize/codegen/vectorize_map_dispatcher.h"

#include "sdfg/targets/vectorize/schedule.h"

#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace vectorize {

VectorizeMapDispatcher::VectorizeMapDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void VectorizeMapDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    // Generate code
    main_stream << "// Map" << std::endl;
    main_stream << "#pragma clang loop vectorize(enable) interleave(enable)" << std::endl;

    main_stream << "for";
    main_stream << "(";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.init());
    main_stream << ";";
    main_stream << language_extension_.expression(node_.condition());
    main_stream << ";";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.update());
    main_stream << ")" << std::endl;
    main_stream << "{" << std::endl;

    main_stream.setIndent(main_stream.indent() + 4);
    codegen::SequenceDispatcher
        dispatcher(language_extension_, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

codegen::InstrumentationInfo VectorizeMapDispatcher::instrumentation_info() const {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    analysis::LoopInfo loop_info = loop_analysis.loop_info(&node_);

    // Perform FlopAnalysis
    std::unordered_map<std::string, std::string> metrics;
    auto& flop_analysis = analysis_manager_.get<analysis::FlopAnalysis>();
    auto flop = flop_analysis.get_if_available_for_codegen(&node_);
    if (!flop.is_null()) {
        std::string flop_str = language_extension_.expression(flop);
        metrics.insert({"flop", flop_str});
    }

    return codegen::InstrumentationInfo(
        node_.element_id(), codegen::ElementType_Map, codegen::TargetType_CPU_PARALLEL, loop_info, metrics
    );
};

} // namespace vectorize
} // namespace sdfg
