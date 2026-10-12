#include "sdfg/codegen/dispatchers/if_else_dispatcher.h"

#include "sdfg/codegen/dispatchers/sequence_dispatcher.h"

namespace sdfg {
namespace codegen {

IfElseDispatcher::IfElseDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::IfElse& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void IfElseDispatcher::dispatch_node(
    PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
) {
    for (size_t i = 0; i < node_.size(); i++) {
        auto child = node_.at(i);

        auto body = language_extension_.begin_branch(main_stream, i, node_.size(), child.second);
        SequenceDispatcher dispatcher(
            language_extension_, sdfg_, analysis_manager_, child.first, instrumentation_plan_, arg_capture_plan_
        );
        dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
        language_extension_.end(main_stream, body);
    }
};

} // namespace codegen
} // namespace sdfg
