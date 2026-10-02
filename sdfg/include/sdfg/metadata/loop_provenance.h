#pragma once

#include <charconv>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "sdfg/element.h"

namespace sdfg::metadata {

inline constexpr const char* ORIGINAL_LOOP_ID_METADATA_KEY = "sdfg.original_loop_id.v1";
inline constexpr const char* RPC_OPTIMIZATION_METADATA_KEY = "docc.rpc_optimization.v1";

struct RpcOptimizationMetadata {
    double expected_speedup;
    std::optional<double> vector_distance;
};

inline std::optional<ElementId> original_loop_id(const Element& element) {
    const auto* serialized = element.metadata_if_exists(ORIGINAL_LOOP_ID_METADATA_KEY);
    if (serialized == nullptr) {
        return std::nullopt;
    }
    ElementId id = 0;
    const auto [end, error] = std::from_chars(serialized->data(), serialized->data() + serialized->size(), id);
    if (error != std::errc{} || end != serialized->data() + serialized->size() || id == 0) {
        throw std::runtime_error("Invalid original loop ID element metadata");
    }
    return id;
}

inline void set_original_loop_id(Element& element, std::optional<ElementId> original_id) {
    if (!original_id.has_value()) {
        element.remove_metadata(ORIGINAL_LOOP_ID_METADATA_KEY);
        return;
    }
    if (original_id.value() == 0) {
        throw std::invalid_argument("Original loop ID must be nonzero");
    }
    element.add_metadata(ORIGINAL_LOOP_ID_METADATA_KEY, std::to_string(original_id.value()));
}

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
    Element& element,
    std::optional<double> expected_speedup,
    std::optional<double> vector_distance = std::nullopt
) {
    if (!expected_speedup.has_value()) {
        element.remove_metadata(RPC_OPTIMIZATION_METADATA_KEY);
        return;
    }
    if (!std::isfinite(expected_speedup.value()) ||
        (vector_distance.has_value() && !std::isfinite(vector_distance.value()))) {
        throw std::invalid_argument("RPC optimization metadata values must be finite");
    }
    nlohmann::json json = {{"expected_speedup", expected_speedup.value()}};
    if (vector_distance.has_value()) {
        json["vector_distance"] = vector_distance.value();
    }
    element.add_metadata(RPC_OPTIMIZATION_METADATA_KEY, json.dump());
}

} // namespace sdfg::metadata
