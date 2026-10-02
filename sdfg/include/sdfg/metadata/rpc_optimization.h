#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "sdfg/element.h"

namespace sdfg::metadata {

inline constexpr const char* RPC_OPTIMIZATION_METADATA_KEY = "docc.rpc_optimization.v1";

struct RpcOptimizationMetadata {
    double expected_speedup;
    std::optional<double> vector_distance;
};

inline std::optional<RpcOptimizationMetadata> rpc_optimization(const Element& element) {
    const auto* serialized = element.metadata_if_exists(RPC_OPTIMIZATION_METADATA_KEY);
    if (serialized == nullptr) {
        return std::nullopt;
    }
    const auto json = nlohmann::json::parse(*serialized);
    if (!json.is_object() || !json.contains("expected_speedup") || !json["expected_speedup"].is_number() ||
        !std::isfinite(json["expected_speedup"].get<double>())) {
        throw std::runtime_error("Invalid RPC optimization element metadata");
    }
    RpcOptimizationMetadata result{json["expected_speedup"].get<double>(), std::nullopt};
    if (json.contains("vector_distance")) {
        if (!json["vector_distance"].is_number() || !std::isfinite(json["vector_distance"].get<double>())) {
            throw std::runtime_error("Invalid RPC vector distance element metadata");
        }
        result.vector_distance = json["vector_distance"].get<double>();
    }
    return result;
}

inline void set_rpc_optimization(
    Element& element, std::optional<double> expected_speedup, std::optional<double> vector_distance = std::nullopt
) {
    if (!expected_speedup.has_value()) {
        element.remove_metadata(RPC_OPTIMIZATION_METADATA_KEY);
        return;
    }
    if (!std::isfinite(expected_speedup.value()) ||
        (vector_distance.has_value() && !std::isfinite(vector_distance.value()))) {
        throw std::invalid_argument("RPC optimization values must be finite");
    }
    nlohmann::json json = {{"expected_speedup", expected_speedup.value()}};
    if (vector_distance.has_value()) {
        json["vector_distance"] = vector_distance.value();
    }
    element.add_metadata(RPC_OPTIMIZATION_METADATA_KEY, json.dump());
}

} // namespace sdfg::metadata
