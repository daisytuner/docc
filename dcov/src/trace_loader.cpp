#include "sdfg/dcov/trace_loader.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <regex>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace sdfg {
namespace dcov {

namespace {

/// Parse the aggregated/per-invocation trace and index profiles by element_id.
/// The last event seen for a given element_id wins (aggregated traces emit one
/// event per region, so collisions are not expected in practice).
std::map<size_t, RuntimeProfile> index_profiles(const nlohmann::json& trace) {
    std::map<size_t, RuntimeProfile> by_eid;

    auto events_it = trace.find("traceEvents");
    if (events_it == trace.end() || !events_it->is_array()) return by_eid;

    for (const auto& ev : *events_it) {
        const auto args_it = ev.find("args");
        if (args_it == ev.end() || !args_it->is_object()) continue;
        const auto docc_it = args_it->find("docc");
        if (docc_it == args_it->end() || !docc_it->is_object()) continue;
        const auto eid_it = docc_it->find("element_id");
        if (eid_it == docc_it->end() || !eid_it->is_number()) continue;

        RuntimeProfile prof;
        prof.runtime_us = ev.value("dur", 0.0);
        prof.target_type = args_it->value("target_type", std::string());

        const auto metrics_it = args_it->find("metrics");
        if (metrics_it != args_it->end() && metrics_it->is_object()) {
            for (const auto& [name, stat] : metrics_it->items()) {
                MetricStat ms;
                ms.name = name;
                if (stat.is_object()) {
                    ms.mean = stat.value("mean", 0.0);
                    ms.min = stat.value("min", 0.0);
                    ms.max = stat.value("max", 0.0);
                    ms.variance = stat.value("variance", 0.0);
                    ms.count = stat.value("count", static_cast<uint64_t>(0));
                } else if (stat.is_number()) {
                    // Per-invocation traces store a bare scalar metric value.
                    ms.mean = ms.min = ms.max = stat.get<double>();
                    ms.count = 1;
                }
                if (ms.name == "runtime") prof.invocations = ms.count;
                prof.metrics.push_back(std::move(ms));
            }
        }

        by_eid[eid_it->get<size_t>()] = std::move(prof);
    }
    return by_eid;
}

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) throw std::runtime_error("cannot open trace '" + path.string() + "'");
    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to parse trace '" + path.string() + "': " + e.what());
    }
    return j;
}

} // namespace

size_t annotate_with_trace(Module& module, const std::filesystem::path& trace_path) {
    const nlohmann::json trace = read_json(trace_path);
    const std::map<size_t, RuntimeProfile> by_eid = index_profiles(trace);

    size_t matched = 0;
    for (auto& region : module.regions) {
        if (region.element_id == 0) continue;
        auto it = by_eid.find(region.element_id);
        if (it == by_eid.end()) continue;
        region.profile = it->second;
        ++matched;
    }
    return matched;
}

std::string arg_capture_path_from_trace(const std::filesystem::path& trace_path) {
    const nlohmann::json trace = read_json(trace_path);
    auto events_it = trace.find("traceEvents");
    if (events_it == trace.end() || !events_it->is_array()) return "";
    for (const auto& ev : *events_it) {
        const auto args_it = ev.find("args");
        if (args_it == ev.end()) continue;
        const auto docc_it = args_it->find("docc");
        if (docc_it == args_it->end()) continue;
        const auto p = docc_it->find("arg_capture_path");
        if (p != docc_it->end() && p->is_string()) return p->get<std::string>();
    }
    return "";
}

size_t annotate_with_arg_captures(Module& module, const std::filesystem::path& arg_capture_dir) {
    if (arg_capture_dir.empty() || !std::filesystem::is_directory(arg_capture_dir)) return 0;

    // Filename: <name>_inv<inv>_arg<idx>_<in|out>_<element_id>.bin
    static const std::regex pattern(R"(_inv\d+_arg(\d+)_(in|out)_(\d+)\.bin$)");

    std::map<size_t, std::vector<std::string>> by_eid;
    for (const auto& entry : std::filesystem::directory_iterator(arg_capture_dir)) {
        if (!entry.is_regular_file()) continue;
        const std::string fname = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_search(fname, m, pattern)) continue;
        const size_t eid = static_cast<size_t>(std::stoull(m[3].str()));
        by_eid[eid].push_back("arg" + m[1].str() + ":" + m[2].str());
    }

    for (auto& [eid, args] : by_eid) {
        std::sort(args.begin(), args.end());
        args.erase(std::unique(args.begin(), args.end()), args.end());
    }

    size_t matched = 0;
    for (auto& region : module.regions) {
        if (region.element_id == 0) continue;
        auto it = by_eid.find(region.element_id);
        if (it == by_eid.end()) continue;
        region.has_arg_capture = true;
        region.arg_captures = it->second;
        ++matched;
    }
    return matched;
}

} // namespace dcov
} // namespace sdfg
