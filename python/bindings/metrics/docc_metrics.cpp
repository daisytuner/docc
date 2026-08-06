#include "docc_metrics.h"
#include <filesystem>
#include <fstream>

namespace docc::metrics {

void DoccMetrics::add_target_options(const target::TargetOptions& target_options) {
    add_metric("target", target_options.target, "target");
    add_metric("category", target_options.category, "target");
    add_metric("remote_tuning", target_options.remote_tuning ? "true" : "false", "target");
}

void DoccMetrics::add_frontend_source_info(const std::string& frontend) { add_metric("frontend", frontend, "source"); }

void DoccMetrics::capture_env_vars() {
    if (auto run_stage_env = std::getenv("DAISY_CI_STAGE")) {
        add_metric("ci_stage", run_stage_env, "source");
    }
    if (auto run_name_env = std::getenv("DAISY_CI_RUN_NAME")) {
        add_metric("ci_run_name", run_name_env, "source");
    }
    if (auto docc_ci_env = std::getenv("DOCC_CI")) {
        add_metric("ci_level", docc_ci_env, "source");
    }
}

void DoccMetrics::add_metric(const std::string& key, const std::string& value, const std::string& section) {
    if (section.empty()) {
        global_.emplace_back(key, value);
        return;
    }
    for (auto& sec : sections_) {
        if (sec.first == section) {
            sec.second.emplace_back(key, value);
            return;
        }
    }
    sections_.push_back({section, {{key, value}}});
}

std::string DoccMetrics::append_to(const std::string& output_dir, const std::string& file_name) const {
    std::filesystem::path dir(output_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create directory '" + output_dir + "': " + ec.message());
    }
    std::filesystem::path path = dir / file_name;
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out) {
        throw std::runtime_error("Failed to open metrics file for writing: " + path.string());
    }
    for (const auto& kv : global_) {
        out << kv.first << "=" << kv.second << "\n";
    }
    for (const auto& sec : sections_) {
        out << "[" << sec.first << "]\n";
        for (const auto& kv : sec.second) {
            out << kv.first << "=" << kv.second << "\n";
        }
    }
    return path.string();
}

} // namespace docc::metrics
