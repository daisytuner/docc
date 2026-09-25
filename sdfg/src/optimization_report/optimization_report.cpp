#include "sdfg/optimization_report/optimization_report.h"

#include <nlohmann/json_fwd.hpp>
#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_sdfg.h"


namespace sdfg {

OptimizationReport::OptimizationReport(StructuredSDFG& sdfg, bool aggregate)
    : sdfg_(sdfg), report_(nlohmann::json::object()), aggregate_(aggregate) {
    analysis::AnalysisManager manager(sdfg);
    auto& loop_analysis = manager.get<analysis::LoopAnalysis>();

    // Add outermost loops
    report_["regions"] = nlohmann::json::array();
    report_["type"] = aggregate ? "aggregate" : "detailed";

    for (int i = 0; i < loop_analysis.outermost_loops().size(); i++) {
        nlohmann::json loop_json;
        auto loop = loop_analysis.outermost_loops().at(i);
        loop_json["loopnest_index"] = i;
        loop_json["element_id"] = loop->element_id();
        serializer::JSONSerializer serializer;
        nlohmann::json debug_info;
        serializer.debug_info_to_json(debug_info, loop->debug_info());
        loop_json["debug_info"] = debug_info;

        if (auto while_loop = dynamic_cast<const structured_control_flow::While*>(loop)) {
            loop_json["type"] = "while";
        } else if (auto for_loop = dynamic_cast<const structured_control_flow::For*>(loop)) {
            loop_json["type"] = "for";
        } else if (auto map = dynamic_cast<const structured_control_flow::Map*>(loop)) {
            loop_json["type"] = "map";
        }
        report_["regions"].push_back(loop_json);
    }
}


nlohmann::json OptimizationReport::get_report() {
    return report_;
}

void OptimizationReport::add_pass_entry(const std::string& pass_name, long duration, bool applied) {
    /* if (!aggregate_) {
        if (!report_.contains("passes")) {
            report_["passes"] = nlohmann::json::array();
        }
        nlohmann::json pass_json;
        pass_json["name"] = pass_name;
        pass_json["duration_us"] = duration;
        pass_json["applied"] = applied;
        report_["passes"].push_back(pass_json);
    } else {
        if (!report_.contains("passes")) {
            report_["passes"] = nlohmann::json::object();
        }
        if (!report_["passes"].contains(pass_name)) {
            report_["passes"][pass_name] = nlohmann::json::object();
            report_["passes"][pass_name]["count"] = 0;
            report_["passes"][pass_name]["total_duration_us"] = 0;
        }
        report_["passes"][pass_name]["count"] = report_["passes"][pass_name]["count"].get<int>() + 1;
        report_["passes"][pass_name]["total_duration_us"] =
    report_["passes"][pass_name]["total_duration_us"].get<long>() + duration;
    } */
}

void OptimizationReport::add_transformation_entry(
    int loopnest_index, const std::string& transformation_name, long apply_duration, const TransformReport& report
) {
    auto& rep = report_["regions"].at(loopnest_index
    )["transformations"][transformation_name] = nlohmann::json::object();

    rep["possible"] = report.possible;
    rep["applied"] = report.applied;
    if (!report.reason.empty()) {
        rep["reason"] = report.reason;
    }
    if (!report.info.empty()) {
        rep["info"] = report.info;
    }

    /* if (!aggregate_) {
        if (!report_.contains("transformations")) {
            report_["transformations"] = nlohmann::json::array();
        }
        nlohmann::json transformation_json;
        transformation_json["name"] = transformation_name;
        transformation_json["duration_us"] = apply_duration;
        transformation_json["description"] = transformation_desc;
        report_["transformations"].push_back(transformation_json);
    } else {
        if (!report_.contains("transformations")) {
            report_["transformations"] = nlohmann::json::object();
        }
        if (!report_["transformations"].contains(transformation_name)) {
            report_["transformations"][transformation_name] = nlohmann::json::object();
            report_["transformations"][transformation_name]["count"] = 0;
            report_["transformations"][transformation_name]["total_duration_us"] = 0;
        }
        report_["transformations"][transformation_name]["count"] =
    report_["transformations"][transformation_name]["count"].get<int>() + 1;
        report_["transformations"][transformation_name]["total_duration_us"] =
    report_["transformations"][transformation_name]["total_duration_us"].get<long>() + apply_duration;
    } */
}

void OptimizationReport::add_target_test(size_t loopnest_index, const std::string& target_name, bool success) {
    if (!report_["regions"].at(loopnest_index).contains("targets")) {
        report_["regions"].at(loopnest_index)["targets"] = nlohmann::json::object();
    }
    report_["regions"].at(loopnest_index)["targets"][target_name] = success;
}

} // namespace sdfg
