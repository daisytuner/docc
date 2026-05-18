#include "docc/lifting/lift_report.h"

#include "sdfg/serializer/json_serializer.h"

namespace docc {
namespace lifting {

static LiftingReport instance;

LiftingReport::LiftingReport() {
    const char* env_var = std::getenv("DOCC_LIFTING_REPORT");
    if (env_var && (std::string(env_var) == "1" || std::string(env_var) == "true" || std::string(env_var) == "yes")) {
        this->do_reporting_ = true;
        this->report_ = nlohmann::json::object();
        this->report_["lifts"] = nlohmann::json::array();
        this->report_["failures"] = nlohmann::json::array();
    } else {
        this->do_reporting_ = false;
    }
}

void LiftingReport::add_successful_lift_internal(const sdfg::DebugInfo& dbg_info) {
    if (!this->do_reporting_) {
        return;
    }
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json dbg_info_json;
    serializer.debug_info_to_json(dbg_info_json, dbg_info);
    nlohmann::json entry;
    entry["debug_info"] = dbg_info_json;

    this->report_["lifts"].push_back(entry);
}

void LiftingReport::
    add_failed_lift_internal(const sdfg::DebugInfo& dbg_info, const std::string& reason, const std::string& element) {
    if (!this->do_reporting_) {
        return;
    }
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json dbg_info_json;
    serializer.debug_info_to_json(dbg_info_json, dbg_info);
    nlohmann::json entry;
    entry["debug_info"] = dbg_info_json;
    entry["reason"] = reason;
    entry["element"] = element;

    this->report_["failures"].push_back(entry);
}

void LiftingReport::dump_report_internal(std::filesystem::path report_path) {
    if (!this->do_reporting_) {
        return;
    }
    std::ofstream report_file(report_path);
    if (!report_file.is_open()) {
        throw std::runtime_error("Failed to open lifting report file: " + report_path.string());
    }
    report_file << this->report_.dump(4);
    report_file.close();
}

void LiftingReport::add_successful_lift(const sdfg::DebugInfo& dbg_info) {
    instance.add_successful_lift_internal(dbg_info);
}

void LiftingReport::add_failed_lift(const sdfg::DebugInfo& dbg_info, const std::string& reason, const std::string& element) {
    instance.add_failed_lift_internal(dbg_info, reason, element);
}

void LiftingReport::dump_report(std::filesystem::path report_path) { instance.dump_report_internal(report_path); }

} // namespace lifting
} // namespace docc
