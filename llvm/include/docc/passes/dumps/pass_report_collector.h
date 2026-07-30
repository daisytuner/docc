#pragma once

#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <llvm/IR/Module.h>
#include "docc/analysis/analysis.h"
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

class PassReportManager : public docc::analysis::Analysis {
private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<PassReportCollector>> collectors_;

protected:
    void run(docc::analysis::AnalysisManager& AM) override {}

public:
    PassReportCollector& get_collector(const llvm::Module& module) {
        auto module_name = module.getName().str();
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = collectors_.find(module_name);
        if (it == collectors_.end()) {
            it = collectors_.emplace(module_name, std::make_unique<PassReportCollector>()).first;
        }
        return *it->second;
    }

    PassReportCollector* get_collector_if_exists(const llvm::Module& module) {
        auto module_name = module.getName().str();
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = collectors_.find(module_name);
        if (it == collectors_.end()) {
            return nullptr;
        } else {
            return it->second.get();
        }
    }
};

} // namespace docc::passes
