#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "sdfg/function.h"

namespace sdfg {
namespace passes::rpc {

inline constexpr const char* LOOP_PROVENANCE_METADATA_KEY = "sdfg.loop_provenance.v1";
inline constexpr const char* RPC_LOOP_RESULTS_METADATA_KEY = "docc.rpc_loop_results.v1";
using LoopProvenanceMap = std::map<ElementId, ElementId>;

inline LoopProvenanceMap read_loop_provenance(const Function& function) {
    const auto* serialized = function.metadata_if_exists(LOOP_PROVENANCE_METADATA_KEY);
    if (serialized == nullptr) {
        return {};
    }
    const auto json = nlohmann::json::parse(*serialized);
    if (!json.is_object()) {
        throw std::runtime_error("SDFG loop provenance metadata must be a JSON object");
    }
    LoopProvenanceMap provenance;
    for (const auto& [key, value] : json.items()) {
        ElementId output_id = 0;
        const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), output_id);
        if (error != std::errc{} || end != key.data() + key.size() || output_id == 0 ||
            (!value.is_number_unsigned() && (!value.is_number_integer() || value.get<int64_t>() < 0))) {
            throw std::runtime_error("Invalid entry in SDFG loop provenance metadata");
        }
        if (!provenance.emplace(output_id, value.get<ElementId>()).second) {
            throw std::runtime_error("Duplicate numeric output loop ID in SDFG loop provenance metadata");
        }
    }
    return provenance;
}

inline ElementId original_loop_id(const LoopProvenanceMap& provenance, ElementId loop_id) {
    const auto it = provenance.find(loop_id);
    return it == provenance.end() ? loop_id : it->second;
}

/** Copy the loop provenance metadata value from the RPC response SDFG verbatim. */
inline void copy_loop_provenance_metadata(Function& target, const Function& response) {
    const auto* provenance = response.metadata_if_exists(LOOP_PROVENANCE_METADATA_KEY);
    target.remove_metadata(LOOP_PROVENANCE_METADATA_KEY);
    if (provenance != nullptr) {
        target.add_metadata(LOOP_PROVENANCE_METADATA_KEY, *provenance);
    }
}

inline void record_rpc_loop_result(
    Function& function,
    ElementId original_id,
    double expected_speedup,
    double vector_distance
) {
    if (original_id == 0 || !std::isfinite(expected_speedup)) {
        return;
    }

    nlohmann::json results = nlohmann::json::object();
    if (const auto* serialized = function.metadata_if_exists(RPC_LOOP_RESULTS_METADATA_KEY)) {
        results = nlohmann::json::parse(*serialized);
        if (!results.is_object()) {
            throw std::runtime_error("RPC loop results metadata must be a JSON object");
        }
    }

    nlohmann::json entry = {{"expected_speedup", expected_speedup}};
    if (std::isfinite(vector_distance)) {
        entry["vector_distance"] = vector_distance;
    }
    results[std::to_string(original_id)] = std::move(entry);
    function.add_metadata(RPC_LOOP_RESULTS_METADATA_KEY, results.dump());
}

} // namespace passes::rpc
} // namespace sdfg
