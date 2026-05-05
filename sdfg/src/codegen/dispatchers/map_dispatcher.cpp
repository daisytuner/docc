#include "sdfg/codegen/dispatchers/map_dispatcher.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace codegen {

SchedTypeMapDispatcher::SchedTypeMapDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void SchedTypeMapDispatcher::
    dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory) {
    auto dispatcher = MapDispatcherRegistry::instance().get_map_dispatcher(node_.schedule_type().value());
    if (dispatcher) {
        auto dispatcher_ptr =
            dispatcher(language_extension_, sdfg_, analysis_manager_, node_, instrumentation_plan_, arg_capture_plan_);
        dispatcher_ptr->dispatch(main_stream, globals_stream, library_snippet_factory);
    } else {
        throw std::runtime_error("Unsupported map schedule type: " + std::string(node_.schedule_type().value()));
    }
};

SequentialMapDispatcher::SequentialMapDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void SequentialMapDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    main_stream << "// Map" << std::endl;
    main_stream << "#pragma clang loop vectorize(disable) interleave(disable)" << std::endl;
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
    SequenceDispatcher
        dispatcher(language_extension_, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    main_stream << "}" << std::endl;
};

InstrumentationInfo MapDispatcher::instrumentation_info() const {
    auto dispatcher = MapDispatcherRegistry::instance().get_map_dispatcher(node_.schedule_type().value());
    if (dispatcher) {
        auto dispatcher_ptr =
            dispatcher(language_extension_, sdfg_, analysis_manager_, node_, instrumentation_plan_, arg_capture_plan_);
        auto map_dispatcher_ptr = static_cast<MapDispatcher*>(dispatcher_ptr.get());
        return map_dispatcher_ptr->instrumentation_info();
    } else {
        throw std::runtime_error("Unsupported map schedule type: " + std::string(node_.schedule_type().value()));
    }
};

InstrumentationInfo SequentialMapDispatcher::instrumentation_info() const {
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

    return InstrumentationInfo(node_.element_id(), ElementType_Map, TargetType_SEQUENTIAL, loop_info, metrics);
};

} // namespace codegen
} // namespace sdfg
