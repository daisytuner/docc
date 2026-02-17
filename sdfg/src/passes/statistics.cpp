#include "sdfg/passes/statistics.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace sdfg {
namespace passes {

void PassStatistics::add_sdfg_pass(const std::string& name, uint64_t milliseconds) {
    if (!sdfg_count_.contains(name)) {
        sdfg_count_.insert({name, 1});
    } else {
        sdfg_count_[name]++;
    }
    if (!sdfg_time_.contains(name)) {
        sdfg_time_.insert({name, milliseconds});
    } else {
        sdfg_time_[name] += milliseconds;
    }
}

void PassStatistics::add_structured_sdfg_pass(const std::string& name, uint64_t milliseconds) {
    if (!structured_sdfg_count_.contains(name)) {
        structured_sdfg_count_.insert({name, 1});
    } else {
        structured_sdfg_count_[name]++;
    }
    if (!structured_sdfg_time_.contains(name)) {
        structured_sdfg_time_.insert({name, milliseconds});
    } else {
        structured_sdfg_time_[name] += milliseconds;
    }
}

std::string PassStatistics::summary() {
    if (sdfg_count_.empty() && structured_sdfg_count_.empty()) {
        return "";
    }

    auto compare_data_fn = [](const std::tuple<std::string, uint64_t, uint64_t>& a,
                              const std::tuple<std::string, uint64_t, uint64_t>& b) {
        auto [a_name, a_count, a_milliseconds] = a;
        auto [b_name, b_count, b_milliseconds] = b;
        return a_milliseconds > b_milliseconds || (a_milliseconds == b_milliseconds && a_count > b_count) ||
               (a_milliseconds == b_milliseconds && a_count == b_count && a_name < b_name);
    };
    std::stringstream stream;
    stream << "Pass Statistics:" << std::endl;

    std::vector<std::tuple<std::string, uint64_t, uint64_t>> sdfg_data;
    uint64_t sdfg_time_sum = 0;
    for (auto [name, count] : sdfg_count_) {
        if (sdfg_time_.contains(name)) {
            auto milliseconds = sdfg_time_[name];
            sdfg_data.push_back({name, count, milliseconds});
            sdfg_time_sum += milliseconds;
        }
    }
    std::sort(sdfg_data.begin(), sdfg_data.end(), compare_data_fn);

    if (!sdfg_data.empty()) {
        stream << "  SDFG Passes: " << sdfg_time_sum << " ms" << std::endl;
        for (auto [name, count, milliseconds] : sdfg_data) {
            stream << "    " << milliseconds << " ms  " << count << "  " << name << std::endl;
        }
    }

    std::vector<std::tuple<std::string, uint64_t, uint64_t>> structured_sdfg_data;
    uint64_t structured_sdfg_time_sum = 0;
    for (auto [name, count] : structured_sdfg_count_) {
        if (structured_sdfg_time_.contains(name)) {
            auto milliseconds = structured_sdfg_time_[name];
            structured_sdfg_data.push_back({name, count, milliseconds});
            structured_sdfg_time_sum += milliseconds;
        }
    }
    std::sort(structured_sdfg_data.begin(), structured_sdfg_data.end(), compare_data_fn);

    if (!structured_sdfg_data.empty()) {
        stream << "  Structured SDFG Passes: " << structured_sdfg_time_sum << " ms" << std::endl;
        for (auto [name, count, milliseconds] : structured_sdfg_data) {
            stream << "    " << milliseconds << " ms  " << count << "  " << name << std::endl;
        }
    }

    return stream.str();
}

void PipelineStatistics::add_sdfg_pipeline(const std::string& name, uint64_t milliseconds) {
    if (!sdfg_count_.contains(name)) {
        sdfg_count_.insert({name, 1});
    } else {
        sdfg_count_[name]++;
    }
    if (!sdfg_time_.contains(name)) {
        sdfg_time_.insert({name, milliseconds});
    } else {
        sdfg_time_[name] += milliseconds;
    }
}

void PipelineStatistics::add_structured_sdfg_pipeline(const std::string& name, uint64_t milliseconds) {
    if (!structured_sdfg_count_.contains(name)) {
        structured_sdfg_count_.insert({name, 1});
    } else {
        structured_sdfg_count_[name]++;
    }
    if (!structured_sdfg_time_.contains(name)) {
        structured_sdfg_time_.insert({name, milliseconds});
    } else {
        structured_sdfg_time_[name] += milliseconds;
    }
}

std::string PipelineStatistics::summary() {
    if (sdfg_count_.empty() && structured_sdfg_count_.empty()) {
        return "";
    }

    auto compare_data_fn = [](const std::tuple<std::string, uint64_t, uint64_t>& a,
                              const std::tuple<std::string, uint64_t, uint64_t>& b) {
        auto [a_name, a_count, a_milliseconds] = a;
        auto [b_name, b_count, b_milliseconds] = b;
        return a_milliseconds > b_milliseconds || (a_milliseconds == b_milliseconds && a_count > b_count) ||
               (a_milliseconds == b_milliseconds && a_count == b_count && a_name < b_name);
    };
    std::stringstream stream;
    stream << "Pipeline Statistics:" << std::endl;

    std::vector<std::tuple<std::string, uint64_t, uint64_t>> sdfg_data;
    uint64_t sdfg_time_sum = 0;
    for (auto [name, count] : sdfg_count_) {
        if (sdfg_time_.contains(name)) {
            auto milliseconds = sdfg_time_[name];
            sdfg_data.push_back({name, count, milliseconds});
            sdfg_time_sum += milliseconds;
        }
    }
    std::sort(sdfg_data.begin(), sdfg_data.end(), compare_data_fn);

    if (!sdfg_data.empty()) {
        stream << "  SDFG Pipelines: " << sdfg_time_sum << " ms" << std::endl;
        for (auto [name, count, milliseconds] : sdfg_data) {
            stream << "    " << milliseconds << " ms  " << count << "  " << name << std::endl;
        }
    }

    std::vector<std::tuple<std::string, uint64_t, uint64_t>> structured_sdfg_data;
    uint64_t structured_sdfg_time_sum = 0;
    for (auto [name, count] : structured_sdfg_count_) {
        if (structured_sdfg_time_.contains(name)) {
            auto milliseconds = structured_sdfg_time_[name];
            structured_sdfg_data.push_back({name, count, milliseconds});
            structured_sdfg_time_sum += milliseconds;
        }
    }
    std::sort(structured_sdfg_data.begin(), structured_sdfg_data.end(), compare_data_fn);

    if (!structured_sdfg_data.empty()) {
        stream << "  Structured SDFG Pipelines: " << structured_sdfg_time_sum << " ms" << std::endl;
        for (auto [name, count, milliseconds] : structured_sdfg_data) {
            stream << "    " << milliseconds << " ms  " << count << "  " << name << std::endl;
        }
    }

    return stream.str();
}

} // namespace passes
} // namespace sdfg
