#include "sdfg/codegen/dispatchers/node_dispatcher.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"

namespace sdfg {
namespace codegen {

NodeDispatcher::NodeDispatcher(
    LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& node,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan
)
    : node_(node), language_extension_(language_extension), sdfg_(sdfg), analysis_manager_(analysis_manager),
      instrumentation_plan_(instrumentation_plan), arg_capture_plan_(arg_capture_plan) {};

bool NodeDispatcher::begin_node(PrettyPrinter& stream) { return false; };

void NodeDispatcher::end_node(PrettyPrinter& stream, bool applied) {};

InstrumentationInfo NodeDispatcher::instrumentation_info() const {
    return InstrumentationInfo(node_.element_id(), node_.element_type(), TargetType_SEQUENTIAL);
};

void NodeDispatcher::
    dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory) {
    bool applied = begin_node(main_stream);

    if (this->arg_capture_plan_.should_instrument(node_)) {
        this->arg_capture_plan_.begin_instrumentation(node_, main_stream, language_extension_);
    }

    bool should_instrument = this->instrumentation_plan_.should_instrument(node_);
    std::optional<InstrumentationInfo> instrumentation_info;
    if (should_instrument) {
        instrumentation_info = this->instrumentation_info();
        this->instrumentation_plan_
            .begin_instrumentation(node_, main_stream, language_extension_, instrumentation_info.value());
    }

    dispatch_node(main_stream, globals_stream, library_snippet_factory);

    if (should_instrument) {
        this->instrumentation_plan_
            .end_instrumentation(node_, main_stream, language_extension_, instrumentation_info.value());
    }

    if (this->arg_capture_plan_.should_instrument(node_)) {
        this->arg_capture_plan_.end_instrumentation(node_, main_stream, language_extension_);
    }

    end_node(main_stream, applied);
};

} // namespace codegen
} // namespace sdfg
