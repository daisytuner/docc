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
    size_t i = 0;
    while (i < node_.size()) {
        auto& child = node_.at(i);

        const auto* group = instrumentation_plan_.group_span_starting_at(child);
        if (group != nullptr && group->sequence == &node_) {
            auto first_dispatcher = create_dispatcher(
                language_extension_, sdfg_, analysis_manager_, child, instrumentation_plan_, arg_capture_plan_
            );
            auto group_info = first_dispatcher->instrumentation_info_for_group();
            group_info.set_sampling(instrumentation_plan_.sampling());
            group_info.set_logical_region_id(group->original_loop_id);
            group_info.set_original_loop_id(group->original_loop_id);
            group_info.set_expected_speedup(instrumentation_plan_.expected_speedup(child));
            group_info.set_vector_distance(instrumentation_plan_.vector_distance(child));
            std::vector<InstrumentationMemberInfo> members;
            members.reserve(group->members.size());
            for (const auto* member : group->members) {
                const auto& debug_info = member->debug_info();
                members.push_back({
                    member->element_id(),
                    debug_info.filename(),
                    debug_info.function(),
                    debug_info.start_line(),
                    debug_info.start_column(),
                    debug_info.end_line(),
                    debug_info.end_column()
                });
            }
            group_info.set_members(std::move(members));

            // For groups, we begin instrumentation before dispatching the members and end it afterward.
            instrumentation_plan_.begin_instrumentation(child, main_stream, language_extension_, group_info);
            for (size_t member_index = 0; member_index < group->members.size(); ++member_index) {
                const size_t child_index = i + member_index;
                auto& member = node_.at(child_index);
                if (member_index == 0) {
                    first_dispatcher->dispatch(main_stream, globals_stream, library_snippet_factory);
                } else {
                    auto dispatcher = create_dispatcher(
                        language_extension_, sdfg_, analysis_manager_, member, instrumentation_plan_, arg_capture_plan_
                    );
                    dispatcher->dispatch(main_stream, globals_stream, library_snippet_factory);
                }
            }
            instrumentation_plan_.end_instrumentation(child, main_stream, language_extension_, group_info);
            i += group->members.size();
            continue;
        }

        // Single node dispatch
        auto dispatcher = create_dispatcher(
            language_extension_, sdfg_, analysis_manager_, child, instrumentation_plan_, arg_capture_plan_
        );
        dispatcher->dispatch(main_stream, globals_stream, library_snippet_factory);
        ++i;
    }
};

} // namespace codegen
} // namespace sdfg
