#include "printf_map_dispatcher.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/arguments_analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/symbolic/symbolic.h>
#include <string>
#include <vector>
#include "printf_target.h"

namespace sdfg {
namespace printf_target {

PrintfMapDispatcher::PrintfMapDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : codegen::NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {}

void PrintfMapDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    // Analyze arguments and locals
    analysis::AnalysisManager analysis_manager(sdfg_);
    analysis::ArgumentsAnalysis& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    auto& used_arguments = arguments_analysis.arguments(analysis_manager, node_);
    auto& locals = arguments_analysis.locals(analysis_manager, node_);


    auto indvar = node_.indvar();
    auto num_iterations = node_.num_iterations();

    // Collect argument names
    std::vector<std::string> argument_names;
    for (auto& argument : used_arguments) {
        argument_names.push_back(argument.first);
    }
    std::sort(argument_names.begin(), argument_names.end());

    // Generate printf for map entry
    main_stream << "printf(\"[PRINTF_TARGET] Entering map (element_id=%zu)\\n\", (size_t)" << node_.element_id() << ");"
                << std::endl;
    main_stream << "printf(\"[PRINTF_TARGET]   Indvar: %s\\n\", \"" << indvar->get_name() << "\");" << std::endl;
    main_stream << "printf(\"[PRINTF_TARGET]   Iterations: %s\\n\", \""
                << this->language_extension_.expression(num_iterations) << "\");" << std::endl;

    // Print arguments
    main_stream << "printf(\"[PRINTF_TARGET]   Arguments: ";
    for (size_t i = 0; i < argument_names.size(); ++i) {
        main_stream << argument_names[i];
        if (i < argument_names.size() - 1) {
            main_stream << ", ";
        }
    }
    main_stream << "\\n\");" << std::endl;

    // Generate the actual loop with printf tracing
    main_stream << "for (long " << indvar->get_name() << " = 0; " << indvar->get_name() << " < "
                << this->language_extension_.expression(num_iterations) << "; ++" << indvar->get_name() << ") {"
                << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    // Print iteration info (only for first few iterations to avoid spam)
    main_stream << "if (" << indvar->get_name() << " < 3 || " << indvar->get_name()
                << " == " << this->language_extension_.expression(num_iterations) << " - 1) {" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);
    main_stream << "printf(\"[PRINTF_TARGET]   Iteration %s = %ld\\n\", \"" << indvar->get_name() << "\", (long)"
                << indvar->get_name() << ");" << std::endl;
    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "} else if (" << indvar->get_name() << " == 3) {" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);
    main_stream << "printf(\"[PRINTF_TARGET]   ... (iterations 3 to %ld omitted)\\n\", (long)("
                << this->language_extension_.expression(num_iterations) << " - 2));" << std::endl;
    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;

    // Dispatch the body
    dispatch_printf_body(main_stream, library_snippet_factory, globals_stream);

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;

    // Generate printf for map exit
    main_stream << "printf(\"[PRINTF_TARGET] Exiting map (element_id=%zu)\\n\", (size_t)" << node_.element_id() << ");"
                << std::endl;
}

void PrintfMapDispatcher::dispatch_printf_body(
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    codegen::PrettyPrinter& globals_stream
) {
    // Dispatch the actual body content
    codegen::SequenceDispatcher
        dispatcher(language_extension_, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_);
    dispatcher.dispatch(stream, globals_stream, library_snippet_factory);
}

codegen::InstrumentationInfo PrintfMapDispatcher::instrumentation_info() const {
    return codegen::
        InstrumentationInfo(node_.element_id(), codegen::ElementType_Map, TargetType_Printf, analysis::LoopInfo{}, {});
}

} // namespace printf_target
} // namespace sdfg
