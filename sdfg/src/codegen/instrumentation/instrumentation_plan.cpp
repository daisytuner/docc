#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace codegen {

void InstrumentationPlan::update(const Element& element, InstrumentationEventType event_type) {
    this->nodes_[&element] = event_type;
}

bool InstrumentationPlan::should_instrument(const Element& node) const { return this->nodes_.count(&node); }

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


    std::string region_uuid = sdfg_name + "_" + std::to_string(node.element_id());

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
    stream << metadata_var << ".element_type = \"" << info.element_desc() << "\";" << std::endl;
    stream << metadata_var << ".target_type = \"" << info.target_type().value() << "\";" << std::endl;

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

    // Initialize region
    switch (this->nodes_.at(&node)) {
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

    // Enter region
    stream << "__daisy_instrumentation_enter(" << region_id_var << ");" << std::endl;
}

void InstrumentationPlan::end_instrumentation(
    const Element& node, PrettyPrinter& stream, LanguageExtension& language_extension, const InstrumentationInfo& info
) const {
    std::string region_id_var = sdfg_.name() + "_" + std::to_string(node.element_id()) + "_id";

    // Exit region
    if (this->nodes_.at(&node) == InstrumentationEventType::CPU) {
        stream << "__daisy_instrumentation_exit(" << region_id_var << ");" << std::endl;
    } else {
        stream << "__daisy_instrumentation_exit(" << region_id_var << ");" << std::endl;
    }

    for (auto entry : info.metrics()) {
        stream << "__daisy_instrumentation_metric(" << region_id_var << ", \"" << entry.first << "\", " << entry.second
               << ");" << std::endl;
    }

    // Finalize region
    stream << "__daisy_instrumentation_finalize(" << region_id_var << ");" << std::endl;
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::none(StructuredSDFG& sdfg) {
    return std::make_unique<InstrumentationPlan>(sdfg, std::unordered_map<const Element*, InstrumentationEventType>{});
}

std::unique_ptr<InstrumentationPlan> InstrumentationPlan::outermost_loops_plan(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& loop_tree_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto ols = loop_tree_analysis.outermost_loops();

    std::unordered_map<const Element*, InstrumentationEventType> nodes;
    for (size_t i = 0; i < ols.size(); i++) {
        auto& loop = ols[i];
        if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(loop)) {
            if (map_node->schedule_type().value() == "CUDA") {
                nodes.insert({loop, InstrumentationEventType::CUDA});
                continue;
            }
        }
        nodes.insert({loop, InstrumentationEventType::CPU}); // Default to CPU if not CUDA
    }

    std::cout << "Created instrumentation plan for " << nodes.size() << " nodes." << std::endl;

    return std::make_unique<InstrumentationPlan>(sdfg, nodes);
}

} // namespace codegen
} // namespace sdfg
