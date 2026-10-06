#pragma once

#include <charconv>
#include <optional>
#include <stdexcept>
#include <string>

#include "sdfg/element.h"

namespace sdfg::metadata {

inline constexpr const char* SOURCE_LOOP_ID_METADATA_KEY = "sdfg.source_loop_id.v1";

inline std::optional<ElementId> source_loop_id(const Element& element) {
    const auto* serialized = element.metadata_if_exists(SOURCE_LOOP_ID_METADATA_KEY);
    if (serialized == nullptr) {
        return std::nullopt;
    }
    ElementId id = 0;
    const auto [end, error] = std::from_chars(serialized->data(), serialized->data() + serialized->size(), id);
    if (error != std::errc{} || end != serialized->data() + serialized->size() || id == 0) {
        throw std::runtime_error("Invalid source loop ID element metadata");
    }
    return id;
}

inline void set_source_loop_id(Element& element, std::optional<ElementId> original_id) {
    if (!original_id.has_value()) {
        element.remove_metadata(SOURCE_LOOP_ID_METADATA_KEY);
        return;
    }
    if (original_id.value() == 0) {
        throw std::invalid_argument("Source loop ID must be nonzero");
    }
    element.add_metadata(SOURCE_LOOP_ID_METADATA_KEY, std::to_string(original_id.value()));
}

} // namespace sdfg::metadata
