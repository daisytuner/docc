#include "sdfg/targets/omp/codegen/omp_map_dispatcher.h"

#include "sdfg/targets/omp/schedule.h"

#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace omp {

OMPMapDispatcher::OMPMapDispatcher(
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

void OMPMapDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    // Mark written locals as private
    analysis::AnalysisManager analysis_manager(sdfg_);
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView users_view(users, node_.root());

    std::vector<std::string> locals;
    for (auto& entry : users.locals(node_.root())) {
        if (users_view.writes(entry).size() > 0 || users_view.moves(entry).size() > 0) {
            locals.push_back(entry);
        }
    }

    // Generate code
    main_stream << "// Map" << std::endl;
    main_stream << "#pragma omp parallel for";

    main_stream << " schedule(";
    if (ScheduleType_OMP::omp_schedule(node_.schedule_type()) == OpenMPSchedule::Static) {
        main_stream << "static)";
    } else if (ScheduleType_OMP::omp_schedule(node_.schedule_type()) == OpenMPSchedule::Dynamic) {
        main_stream << "dynamic)";
    } else if (ScheduleType_OMP::omp_schedule(node_.schedule_type()) == OpenMPSchedule::Guided) {
        main_stream << "guided)";
    } else {
        throw std::runtime_error("Unsupported OpenMP schedule type");
    }

    if (!ScheduleType_OMP::num_threads(node_.schedule_type()).is_null()) {
        main_stream << " num_threads(";
        main_stream << language_extension_.expression(ScheduleType_OMP::num_threads(node_.schedule_type()));
        main_stream << ")";
    }

    if (locals.size() > 0) {
        main_stream << " private(" << helpers::join(locals, ", ") << ")";
    }
    main_stream << std::endl;
    main_stream << "for";
    main_stream << "(";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.init());
    main_stream << ";";

    // Default condition
    auto condition = node_.condition();

    // OpenMP requires the loop condition to be in a simple form (no && or ||), so we rely on canonical_bound()
    auto canonical_bound = node_.canonical_bound_upper();
    if (!canonical_bound.is_null()) {
        condition = symbolic::Lt(node_.indvar(), canonical_bound);
    }
    main_stream << language_extension_.expression(condition);
    main_stream << ";";
    main_stream << node_.indvar()->get_name();
    main_stream << " = ";
    main_stream << language_extension_.expression(node_.update());
    main_stream << ")" << std::endl;
    main_stream << "{" << std::endl;

    for (auto& local : locals) {
        auto& type = sdfg_.type(local);
        if (type.storage_type().allocation() == types::StorageType::AllocationType::Managed) {
            if (type.storage_type().is_cpu_stack()) {
                main_stream << local << " = ";
                main_stream << "alloca(" << language_extension_.expression(type.storage_type().allocation_size())
                            << ")";
                main_stream << ";" << std::endl;
            } else if (type.storage_type().is_cpu_heap()) {
                main_stream << local << " = ";
                main_stream << language_extension_.external_prefix() << "malloc("
                            << language_extension_.expression(type.storage_type().allocation_size()) << ")";
                main_stream << ";" << std::endl;
            }
        }
    }

    main_stream.setIndent(main_stream.indent() + 4);
    codegen::SequenceDispatcher
        dispatcher(language_extension_, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_);
    dispatcher.dispatch(main_stream, globals_stream, library_snippet_factory);
    main_stream.setIndent(main_stream.indent() - 4);

    for (auto& local : locals) {
        auto& type = sdfg_.type(local);
        if (type.storage_type().deallocation() == types::StorageType::AllocationType::Managed) {
            if (type.storage_type().is_cpu_heap()) {
                main_stream << language_extension_.external_prefix() << "free(" << local << ");" << std::endl;
            }
        }
    }

    main_stream << "}" << std::endl;
};

codegen::InstrumentationInfo OMPMapDispatcher::instrumentation_info() const {
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

} // namespace omp
} // namespace sdfg
