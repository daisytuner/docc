#include "docc/util/cuda_query_compute_capability.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

namespace docc::util {

namespace {

/// Runs a command and captures its standard output. Returns std::nullopt if the
/// command could not be launched.
static std::optional<std::string> _run_and_capture(const std::string& command) {
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), &pclose);
    if (!pipe) {
        return std::nullopt;
    }

    std::string output;
    std::array<char, 256> buffer{};
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        output += buffer.data();
    }
    return output;
}

/// Trims surrounding whitespace from a string.
static std::string _trim(const std::string& raw) {
    const auto begin = raw.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = raw.find_last_not_of(" \t\r\n");
    return raw.substr(begin, end - begin + 1);
}

/// Parses a compute capability string like "8.6" into the integer form clang
/// uses (86). "12.0" becomes 120. Returns std::nullopt on malformed input.
static std::optional<uint32_t> _parse_compute_cap(const std::string& raw) {
    const std::string trimmed = _trim(raw);
    if (trimmed.empty()) {
        return std::nullopt;
    }

    const auto dot = trimmed.find('.');
    if (dot == std::string::npos) {
        return std::nullopt;
    }

    const std::string major_str = trimmed.substr(0, dot);
    const std::string minor_str = trimmed.substr(dot + 1);
    if (major_str.empty() || minor_str.empty()) {
        return std::nullopt;
    }

    try {
        const int major = std::stoi(major_str);
        const int minor = std::stoi(minor_str);
        // clang convention: major * 10 + minor (e.g. 8.6 -> 86, 12.0 -> 120).
        return static_cast<uint32_t>(major * 10 + minor);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

} // namespace

std::vector<CudaComputeCapability> query_cuda_compute_capabilities() {
    const auto output = _run_and_capture("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null");
    if (!output) {
        return {};
    }

    // Collect the distinct device names per compute capability. std::map keeps
    // the capabilities ordered so we can emit them descending below.
    std::map<uint32_t, std::vector<std::string>> caps;
    std::istringstream lines(*output);
    std::string line;
    while (std::getline(lines, line)) {
        if (_trim(line).empty()) {
            continue;
        }
        // Output line format is "<name>, <compute_cap>"; the compute cap is the
        // last comma-separated field, the name is everything before it.
        const auto comma = line.find_last_of(',');
        if (comma == std::string::npos) {
            continue;
        }
        const std::string name = _trim(line.substr(0, comma));
        const auto cap = _parse_compute_cap(line.substr(comma + 1));
        if (!cap) {
            continue;
        }

        auto& names = caps[*cap];
        if (std::find(names.begin(), names.end(), name) == names.end()) {
            names.push_back(name);
        }
    }

    // Emit highest to lowest compute capability.
    std::vector<CudaComputeCapability> result;
    result.reserve(caps.size());
    for (auto it = caps.rbegin(); it != caps.rend(); ++it) {
        result.push_back({.compute_cap = it->first, .device_names = std::move(it->second)});
    }
    return result;
}

void clang_21_set_cuda_forward_compatible_options(
    compile::SrcFileCompilerBuilder& builder, compile::SrcFileCompilerBuilder& snippet_builder
) {
    builder.add_compile_option("--cuda-gpu-arch=sm_70");
    builder.add_compile_option("--cuda-include-ptx=all");
    snippet_builder.add_compile_option("--cuda-gpu-arch=sm_70");
    snippet_builder.add_compile_option("--cuda-include-ptx=all");
}

void clang_21_set_cuda_forward_compatible_options(std::vector<std::string>& compiler_args) {
    compiler_args.emplace_back("--cuda-gpu-arch=sm_70");
    compiler_args.emplace_back("--cuda-include-ptx=all");
}

void clang_21_set_cuda_specific_compute_cap(
    compile::SrcFileCompilerBuilder& builder, compile::SrcFileCompilerBuilder& snippet_builder, uint32_t sm_cap
) {
    auto str = std::to_string(sm_cap);
    builder.add_compile_option("--cuda-gpu-arch=sm_" + str);
    snippet_builder.add_compile_option("--cuda-gpu-arch=sm_" + str);
}

void clang_21_set_cuda_specific_compute_cap(std::vector<std::string>& compiler_args, uint32_t sm_cap) {
    auto str = std::to_string(sm_cap);
    compiler_args.emplace_back("--cuda-gpu-arch=sm_" + str);
}

} // namespace docc::util
