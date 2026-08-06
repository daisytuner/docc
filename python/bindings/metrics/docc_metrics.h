#pragma once
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace docc {
namespace metrics {
/**
 * @brief Collects key/value metrics and serializes them to a classic
 *        ".properties" file ("key=value", with optional "[section-name]"
 *        headers).
 *
 * Metrics are stored in insertion order. Metrics without a section are written
 * first (before any section header); metrics belonging to a section are grouped
 * under a "[section]" header in first-seen order.
 */
class DoccMetrics {
public:
    DoccMetrics() = default;
    void add_metric(const std::string& key, const std::string& value, const std::string& section = "") {
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
    std::string append_to(const std::string& output_dir, const std::string& file_name = "docc_metrics.properties")
        const {
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

private:
    std::vector<std::pair<std::string, std::string>> global_;
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> sections_;
};
} // namespace metrics
} // namespace docc
