#pragma once

#include <iostream>
#include <string>

#include "sdfg/optimization_report/optimization_report.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/transformations/transformation.h"

namespace docc::passes {

struct RegionReport {
    std::unordered_map<std::string, sdfg::TransformReport> transform_results;
    std::unordered_map<std::string, bool> targets_possible;
};

class PassReportCollector : public sdfg::PassReportConsumer {
private:
    std::unordered_map<std::string, std::unique_ptr<std::unordered_map<int32_t, std::unique_ptr<RegionReport>>>>
        reports_;
    std::unordered_map<int32_t, std::unique_ptr<RegionReport>>* current_sdfg_;
    RegionReport* current_region_;

    RegionReport& require_current_region();

public:
    void transform_impossible(const std::string& transform, const std::string& reason) override;

    void transform_possible(const std::string& transform) override;

    void transform_applied(const std::string& transform, nlohmann::json transform_info = {}) override;


    void in_scope(sdfg::StructuredSDFG* scope) override;

    void in_outermost_loop(int idx) override;

    void target_transform_possible(const std::string basicString, bool b) override;

    std::unordered_map<int32_t, std::unique_ptr<RegionReport>>* get_scope_reports(sdfg::StructuredSDFG* scope) const;
};

} // namespace docc::passes
