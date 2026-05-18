#include "docc/passes/dumps/pass_report_collector.h"

namespace docc::passes {

RegionReport& docc::passes::PassReportCollector::require_current_region() {
    auto reg = current_region_;
    if (!reg) {
        if (current_sdfg_) {
            auto& holder = (*current_sdfg_)[-1];
            if (!holder) {
                holder = std::make_unique<RegionReport>();
            }
            current_region_ = holder.get();
            reg = holder.get();
        } else {
            throw std::runtime_error("No SDFG scope");
        }
    }
    return *reg;
}

void docc::passes::PassReportCollector::transform_impossible(const std::string& transform, const std::string& reason) {
    auto& holder = require_current_region().transform_results[transform];
    holder.possible = false;
    holder.applied = false;
    holder.reason = reason;
}

void docc::passes::PassReportCollector::transform_possible(const std::string& transform) {
    auto& holder = require_current_region().transform_results[transform];
    holder.possible = true;
}

void docc::passes::PassReportCollector::transform_applied(const std::string& transform, nlohmann::json transform_info) {
    auto& holder = require_current_region().transform_results[transform];
    holder.applied = true;
    holder.info = transform_info;
}

void docc::passes::PassReportCollector::in_scope(sdfg::StructuredSDFG* scope) {
    if (scope) {
        auto& holder = reports_[scope->name()];
        if (!holder) {
            holder = std::make_unique<std::unordered_map<int32_t, std::unique_ptr<RegionReport>>>();
        }
        current_sdfg_ = holder.get();
        current_region_ = nullptr;
    } else {
        current_sdfg_ = nullptr;
        current_region_ = nullptr;
    }
}

void docc::passes::PassReportCollector::in_outermost_loop(int idx) {
    if (idx >= 0) {
        if (current_sdfg_) {
            auto& holder = (*current_sdfg_)[idx];
            if (!holder) {
                holder = std::make_unique<RegionReport>();
            }
            current_region_ = holder.get();
        } else {
            throw std::runtime_error("No SDFG scope");
        }
    } else {
        current_region_ = nullptr;
    }
}

std::unordered_map<int32_t, std::unique_ptr<RegionReport>>* docc::passes::PassReportCollector::
    get_scope_reports(sdfg::StructuredSDFG* scope) const {
    auto it = reports_.find(scope->name());
    if (it != reports_.end()) {
        return it->second.get();
    } else {
        return nullptr;
    }
}

void PassReportCollector::target_transform_possible(const std::string basicString, bool b) {
    require_current_region().targets_possible[basicString] = b;
}

} // namespace docc::passes
