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
    begin_node_applied_ = begin_node(main_stream);

    if (this->arg_capture_plan_.should_instrument(node_)) {
        this->arg_capture_plan_.begin_instrumentation(node_, main_stream, language_extension_);
    }

    if (this->instrumentation_plan_.should_instrument(node_)) {
        auto info = this->instrumentation_info();
        this->instrumentation_plan_.begin_instrumentation(node_, main_stream, language_extension_, info);
    }

    dispatch_node(main_stream, globals_stream, library_snippet_factory);

    end_dispatch(main_stream);
};

void NodeDispatcher::emit_instrumentation_exit(PrettyPrinter& main_stream) {
    if (this->instrumentation_plan_.should_instrument(node_)) {
        auto info = this->instrumentation_info();
        this->instrumentation_plan_.end_instrumentation(node_, main_stream, language_extension_, info);
    }

    if (this->arg_capture_plan_.should_instrument(node_)) {
        this->arg_capture_plan_.end_instrumentation(node_, main_stream, language_extension_);
    }
};

void NodeDispatcher::end_dispatch(PrettyPrinter& main_stream) {
    if (end_dispatched_) {
        return;
    }
    end_dispatched_ = true;

    emit_instrumentation_exit(main_stream);

    end_node(main_stream, begin_node_applied_);
};

} // namespace codegen
} // namespace sdfg
