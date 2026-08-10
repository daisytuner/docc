#pragma once
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/plugins/targets.h"

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

    void add_target_options(const target::TargetOptions& target_options);

    void add_frontend_source_info(const std::string& frontend);

    void capture_env_vars();

    void add_metric(const std::string& key, const std::string& value, const std::string& section = "");

    std::string append_to(const std::string& output_dir, const std::string& file_name = "docc_metrics.properties") const;

private:
    std::vector<std::pair<std::string, std::string>> global_;
    std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>> sections_;
};


} // namespace metrics
} // namespace docc
