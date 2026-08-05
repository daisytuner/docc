#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"

namespace sdfg {
namespace codegen {

SequenceDispatcher::SequenceDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void SequenceDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    for (size_t i = 0; i < node_.size(); i++) {
        auto& child = node_.at(i);

        // Node
        auto dispatcher = create_dispatcher(
            language_extension_, sdfg_, analysis_manager_, child, instrumentation_plan_, arg_capture_plan_
        );
        dispatcher->dispatch(main_stream, globals_stream, library_snippet_factory);
    }
};

} // namespace codegen
} // namespace sdfg
