#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include <algorithm>
#include <charconv>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <nlohmann/json.hpp>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/element.h"
#include "sdfg/metadata/loop_provenance.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace codegen {

namespace {

std::string escape_cpp_string(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char character : value) {
        switch (character) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += character;
                break;
        }
    }
    return escaped;
}

bool contains_function_return(const structured_control_flow::ControlFlowNode& node) {
    if (dyn_cast<const structured_control_flow::Return*>(&node) != nullptr) {
        return true;
    }
    if (const auto* sequence = dyn_cast<const structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            if (contains_function_return(sequence->at(i))) {
                return true;
            }
        }
    } else if (const auto* if_else = dyn_cast<const structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); ++i) {
            if (contains_function_return(if_else->at(i).first)) {
                return true;
            }
        }
    } else if (const auto* loop = dyn_cast<const structured_control_flow::StructuredLoop*>(&node)) {
        return contains_function_return(loop->root());
    } else if (const auto* loop = dyn_cast<const structured_control_flow::While*>(&node)) {
        return contains_function_return(loop->root());
    }
    return false;
}

} // namespace

bool InstrumentationPlan::should_instrument(const Element& node) const {
    return this->nodes_.count(&node);
}

void InstrumentationPlan::begin_instrumentation(
    const Element& node, PrettyPrinter& stream, LanguageExtension& language_extension, const InstrumentationInfo& info
) const {
    auto& metadata = sdfg_.metadata();
    std::string sdfg_name = sdfg_.name();

    std::string sdfg_file;
    if (auto it = metadata.find("sdfg_file"); it != metadata.end()) {
        sdfg_file = it->second;
    } else {
        sdfg_file = "";
    }

    std::string arg_capture_path;
    if (auto it = metadata.find("arg_capture_path"); it != metadata.end()) {
        arg_capture_path = it->second;
    } else {
        arg_capture_path = "";
    }

    std::string features_file;
    if (auto it = metadata.find("features_file"); it != metadata.end()) {
        features_file = it->second;
    } else {
        features_file = "";
    }

    std::string opt_report_file;
    if (auto it = metadata.find("opt_report_file"); it != metadata.end()) {
        opt_report_file = it->second;
    } else {
        opt_report_file = "";
    }

    std::string transfer_tuning_session_id;
    if (auto it = metadata.find("transfer_tuning_session_id"); it != metadata.end()) {
        transfer_tuning_session_id = it->second;
    } else {
        transfer_tuning_session_id = "";
    }

    const auto logical_region_id = info.logical_region_id();
    std::string region_uuid = logical_region_id.has_value()
                                  ? sdfg_name + "_origin_" + std::to_string(logical_region_id.value())
                                  : sdfg_name + "_" + std::to_string(node.element_id());

    // Create region id variable
    std::string region_id_var = sdfg_name + "_" + std::to_string(node.element_id()) + "_id";

    // Create metadata variable
    std::string metadata_var = sdfg_name + "_" + std::to_string(node.element_id()) + "_md";
    stream << "__daisy_metadata_t " << metadata_var << ";" << std::endl;

    // Source metadata
    auto& dbg_info = node.debug_info();
    stream << metadata_var << ".file_name = \"" << dbg_info.filename() << "\";" << std::endl;
    stream << metadata_var << ".function_name = \"" << dbg_info.function() << "\";" << std::endl;
    stream << metadata_var << ".line_begin = " << dbg_info.start_line() << ";" << std::endl;
    stream << metadata_var << ".line_end = " << dbg_info.end_line() << ";" << std::endl;
    stream << metadata_var << ".column_begin = " << dbg_info.start_column() << ";" << std::endl;
    stream << metadata_var << ".column_end = " << dbg_info.end_column() << ";" << std::endl;

    // DOCC Metadata
    stream << metadata_var << ".sdfg_name = \"" << sdfg_name << "\";" << std::endl;
    stream << metadata_var << ".sdfg_file = \"" << sdfg_file << "\";" << std::endl;
    stream << metadata_var << ".arg_capture_path = \"" << arg_capture_path << "\";" << std::endl;
    stream << metadata_var << ".features_file = \"" << features_file << "\";" << std::endl;
    stream << metadata_var << ".opt_report_file = \"" << opt_report_file << "\";" << std::endl;

    // Element metadata
    stream << metadata_var << ".element_id = " << info.element_id() << ";" << std::endl;
    stream << metadata_var << ".source_loop_id = "
           << (info.source_loop_id().has_value() ? std::to_string(info.source_loop_id().value()) : "0") << ";"
           << std::endl;
    stream << metadata_var << ".element_type = \"" << info.element_desc() << "\";" << std::endl;
    stream << metadata_var << ".target_type = \"" << info.target_type().value() << "\";" << std::endl;

    nlohmann::json member_metadata = nlohmann::json::array();
    for (const auto& member : info.members()) {
        member_metadata.push_back(
            {{"element_id", member.element_id},
             {"filename", member.filename},
             {"function", member.function},
             {"start_line", member.start_line},
             {"start_column", member.start_column},
             {"end_line", member.end_line},
             {"end_column", member.end_column}}
        );
    }
    stream << metadata_var << ".member_loops_json = \"" << escape_cpp_string(member_metadata.dump()) << "\";"
           << std::endl;
    stream << metadata_var << ".expected_performance_json = "
           << (info.expected_performance().has_value()
                   ? "\"" + escape_cpp_string(info.expected_performance()->dump()) + "\""
                   : "\"\"")
           << ";" << std::endl;
    stream << metadata_var << ".vector_distance = "
           << (info.vector_distance().has_value() ? std::to_string(info.vector_distance().value()) : "-1.0") << ";"
           << std::endl;

    // Loop info metadata
    stream << metadata_var << ".loopnest_index = " << info.loop_info().loopnest_index << ";" << std::endl;
    stream << metadata_var << ".num_loops = " << info.loop_info().num_loops << ";" << std::endl;
    stream << metadata_var << ".num_maps = " << info.loop_info().num_maps << ";" << std::endl;
    stream << metadata_var << ".num_fors = " << info.loop_info().num_fors << ";" << std::endl;
    stream << metadata_var << ".num_whiles = " << info.loop_info().num_whiles << ";" << std::endl;
    stream << metadata_var << ".max_depth = " << info.loop_info().max_depth << ";" << std::endl;
    stream << metadata_var << ".is_perfectly_nested = " << (info.loop_info().is_perfectly_nested ? "1" : "0") << ";"
           << std::endl;
    stream << metadata_var << ".is_perfectly_parallel = " << (info.loop_info().is_perfectly_parallel ? "1" : "0") << ";"
           << std::endl;
    stream << metadata_var << ".is_elementwise = " << (info.loop_info().is_elementwise ? "1" : "0") << ";" << std::endl;
    stream << metadata_var << ".has_side_effects = " << (info.loop_info().has_side_effects ? "1" : "0") << ";"
           << std::endl;

    stream << metadata_var << ".region_uuid = \"" << region_uuid << "\";" << std::endl;

    stream << metadata_var << ".transfer_tuning_session_id = \"" << transfer_tuning_session_id << "\";" << std::endl;

    // Initialize region
    switch (info.event_type()) {
        case InstrumentationEventType::CPU: {
            stream << "long long " << region_id_var << " = __daisy_instrumentation_init(&" << metadata_var
                   << ", __DAISY_EVENT_SET_CPU);" << std::endl;
            break;
        }
        case InstrumentationEventType::CUDA: {
            stream << "long long " << region_id_var << " = __daisy_instrumentation_init(&" << metadata_var
                   << ", __DAISY_EVENT_SET_CUDA);" << std::endl;
            break;
        }
        case InstrumentationEventType::NONE: {
            stream << "long long " << region_id_var << " = __daisy_instrumentation_init(&" << metadata_var
                   << ", __DAISY_EVENT_SET_NONE);" << std::endl;
            break;
        }
    }

    // Adaptive sampling: repeat the region until its runtime confidence interval
    // converges. The loop opens before enter and closes after exit (see
    // end_instrumentation), so the region body runs once per sample.
    if (info.sampling()) {
        stream << "while (true) {" << std::endl;
        // Cold sampling (no-op unless DOCC_MEASURE_COLD): evict the working set so
        // each sample re-incurs cold-start misses at the L3/DRAM level.
        stream << "__daisy_instrumentation_flush_caches();" << std::endl;
    }

    // Enter region
    stream << "__daisy_instrumentation_enter(" << region_id_var << ");" << std::endl;
}

void InstrumentationPlan::end_instrumentation(
    const Element& node, PrettyPrinter& stream, LanguageExtension& language_extension, const InstrumentationInfo& info
) const {
    std::string region_id_var = sdfg_.name() + "_" + std::to_string(node.element_id()) + "_id";

    // Exit region
    switch (info.event_type()) {
        case InstrumentationEventType::CPU:
        case InstrumentationEventType::CUDA:
        case InstrumentationEventType::NONE:
            stream << "__daisy_instrumentation_exit(" << region_id_var << ");" << std::endl;
            break;
    }

    // Close the adaptive sampling loop: take another sample unless the runtime
    // confidence interval has converged or the sample/time caps are hit.
    if (info.sampling()) {
        stream << "if (!__daisy_instrumentation_should_continue(" << region_id_var << ")) break;" << std::endl;
        stream << "}" << std::endl;
    }

    for (auto entry : info.metrics()) {
        stream << "__daisy_instrumentation_metric(" << region_id_var << ", \"" << entry.first << "\", " << entry.second
               << ");" << std::endl;
    }

    // Finalize region
    stream << "__daisy_instrumentation_finalize(" << region_id_var << ");" << std::endl;
}

void InstrumentationPlan::
    leaving_instrumentation_function(PrettyPrinter& stream, LanguageExtension& language_extension) const {
    if (!this->is_empty() && this->emit_finalize_all_) {
        stream << "__daisy_instrumentation_finalize_all();" << std::endl;
    }
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::none(StructuredSDFG& sdfg) {
    return std::make_unique<InstrumentationPlan>(sdfg, std::unordered_set<const Element*>{});
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::
    outermost_loops_plan(StructuredSDFG& sdfg, bool emit_finalize_all, bool sampling) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_tree_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto ols = loop_tree_analysis.outermost_loops();

    std::unordered_set<const Element*> nodes;
    for (size_t i = 0; i < ols.size(); i++) {
        nodes.insert(ols[i]);
    }

    DEBUG_PRINTLN("Created instrumentation plan for " << nodes.size() << " nodes.");
    return std::make_unique<InstrumentationPlan>(sdfg, nodes, emit_finalize_all, sampling);
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::
    provenance_grouped_outermost_loops_plan(StructuredSDFG& sdfg, bool emit_finalize_all, bool sampling) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    std::unordered_set<const Element*> nodes;
    std::unordered_map<const Element*, ElementId> logical_region_ids;
    std::unordered_map<ElementId, std::vector<const structured_control_flow::ControlFlowNode*>> loops_by_origin;
    for (auto* loop : loop_analysis.outermost_loops()) {
        nodes.insert(loop);
        const auto origin_id = metadata::source_loop_id(*loop);
        if (origin_id.has_value()) {
            logical_region_ids.emplace(loop, origin_id.value());
            loops_by_origin[origin_id.value()].push_back(loop);
        }
    }

    auto plan = std::make_unique<InstrumentationPlan>(sdfg, nodes, emit_finalize_all, sampling);
    for (auto& [origin_id, members] : loops_by_origin) {
        if (members.size() == 1) {
            logical_region_ids.emplace(members.front(), origin_id);
            continue;
        }

        auto* parent_sequence = dyn_cast<const structured_control_flow::Sequence*>(members.front()->get_parent());
        bool eligible = parent_sequence != nullptr;
        for (auto* member : members) {
            eligible = eligible &&
                       dyn_cast<const structured_control_flow::Sequence*>(member->get_parent()) == parent_sequence &&
                       !contains_function_return(*member);
        }
        if (!eligible) {
            continue;
        }

        std::sort(members.begin(), members.end(), [&](const auto* left, const auto* right) {
            return parent_sequence->index(*left) < parent_sequence->index(*right);
        });
        const int first_index = parent_sequence->index(*members.front());
        for (size_t i = 0; i < members.size(); ++i) {
            if (parent_sequence->index(*members[i]) != first_index + static_cast<int>(i)) {
                eligible = false;
                break;
            }
        }
        if (!eligible) {
            continue;
        }

        const size_t span_index = plan->group_spans_.size();
        plan->group_spans_.push_back({origin_id, parent_sequence, members});
        plan->group_start_by_node_.emplace(members.front(), span_index);
        for (auto* member : members) {
            plan->grouped_members_.insert(member);
            logical_region_ids.emplace(member, origin_id);
        }
    }
    plan->logical_region_ids_ = std::move(logical_region_ids);

    DEBUG_PRINTLN(
        "Created provenance-grouped OLS instrumentation plan for " << nodes.size() << " nodes and "
                                                                   << plan->group_spans_.size() << " spans."
    );
    return plan;
}

} // namespace codegen
} // namespace sdfg
