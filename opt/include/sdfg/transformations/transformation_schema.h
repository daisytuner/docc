#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace sdfg {
namespace transformations {

/**
 * @brief The transformation serialization schema as a JSON Schema (2020-12) document.
 *
 * The schema is authored in opt/json/transformation.schema.json and embedded into
 * the library at build time, so it travels with the binary and serves as the
 * canonical, human-readable documentation of the contract enforced below.
 *
 * @return The raw schema document text.
 */
const std::string& transformation_schema_text();

/**
 * @brief The transformation serialization schema, parsed once.
 * @return The parsed JSON Schema document.
 */
const nlohmann::json& transformation_schema();

/**
 * @brief Validate that a transformation description respects the serialization schema.
 *
 * Every transformation's to_json() must produce an object with exactly this shape:
 *  - "transformation_type": non-empty string (equal to Transformation::name()).
 *  - "subgraph": object whose keys are the contiguous indices "0".."N-1". Each
 *    entry is an object with an unsigned "element_id" and a string "type" (the
 *    flat serialization of a constructor node).
 *  - "parameters": optional object (possibly empty) holding additional constructor
 *    inputs. Omitted for backward compatibility with older descriptions.
 *
 * This is the single source of truth for the schema. It is consumed by the
 * serialization tests and by the runtime invariants in the Recorder/Replayer.
 *
 * @param j The transformation description to validate.
 * @param error_out Populated with a human-readable reason when validation fails.
 * @return true if the description matches the schema.
 */
inline bool validate_transformation_schema(const nlohmann::json& j, std::string& error_out) {
    if (!j.is_object()) {
        error_out = "transformation description must be an object";
        return false;
    }

    // transformation_type
    if (!j.contains("transformation_type")) {
        error_out = "missing 'transformation_type'";
        return false;
    }
    if (!j["transformation_type"].is_string() || j["transformation_type"].get<std::string>().empty()) {
        error_out = "'transformation_type' must be a non-empty string";
        return false;
    }

    // parameters (optional, for backward compatibility; must be an object when present)
    if (j.contains("parameters") && !j["parameters"].is_object()) {
        error_out = "'parameters' must be an object";
        return false;
    }

    // subgraph
    if (!j.contains("subgraph")) {
        error_out = "missing 'subgraph'";
        return false;
    }
    const auto& subgraph = j["subgraph"];
    if (!subgraph.is_object()) {
        error_out = "'subgraph' must be an object";
        return false;
    }
    for (std::size_t i = 0; i < subgraph.size(); ++i) {
        const auto key = std::to_string(i);
        if (!subgraph.contains(key)) {
            error_out = "'subgraph' keys must be contiguous indices; missing '" + key + "'";
            return false;
        }
        const auto& node = subgraph[key];
        if (!node.is_object()) {
            error_out = "'subgraph[" + key + "]' must be an object";
            return false;
        }
        if (!node.contains("element_id")) {
            error_out = "'subgraph[" + key + "]' must have an 'element_id'";
            return false;
        }
        const auto& element_id = node["element_id"];
        const bool non_negative_integer = element_id.is_number_unsigned() ||
                                          (element_id.is_number_integer() && element_id.get<long long>() >= 0);
        if (!non_negative_integer) {
            error_out = "'subgraph[" + key + "]' must have a non-negative integer 'element_id'";
            return false;
        }
        if (!node.contains("type") || !node["type"].is_string()) {
            error_out = "'subgraph[" + key + "]' must have a string 'type'";
            return false;
        }
    }

    // Reject unknown top-level keys to keep the schema strict.
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() != "transformation_type" && it.key() != "subgraph" && it.key() != "parameters") {
            error_out = "unexpected top-level key '" + it.key() + "'";
            return false;
        }
    }

    return true;
}

} // namespace transformations
} // namespace sdfg
