#include "sdfg/data_flow/pointer_metadata.h"

#include "cereal/types/utility.hpp"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg::data_flow {

static void
serialize_region(nlohmann::json& entry, const char* key, const std::optional<math::tensor::TensorLayout>& region) {
    if (region) {
        region->serialize_to_json(entry[key]);
    }
}

static std::optional<math::tensor::TensorLayout> deserialize_region(const nlohmann::json& entry, const char* key) {
    auto it = entry.find(key);
    if (it == entry.end() || it->is_null()) {
        return std::nullopt;
    }
    return math::tensor::TensorLayout::deserialize_from_json(*it);
}

// Legacy scalar-size constructors describe a flat 1-D region `{size}`; a null
// size means the access is unbounded. A structured layout, when given, wins.
static std::optional<math::tensor::TensorLayout>
region_from_size(const symbolic::Expression& size, std::optional<math::tensor::TensorLayout> layout) {
    if (layout) {
        return layout;
    }
    if (size.is_null()) {
        return std::nullopt;
    }
    return math::tensor::TensorLayout(symbolic::MultiExpression{size});
}

static void serialize_access_region(nlohmann::json& entry, const AccessRegion& r) {
    entry["may_access"] = r.may_access;
    entry["covers_all"] = r.covers_all;
    serialize_region(entry, "region", r.region);
}

static AccessRegion deserialize_access_region(const nlohmann::json& entry) {
    AccessRegion r;
    r.may_access = entry.at("may_access").get<bool>();
    r.covers_all = entry.at("covers_all").get<bool>();
    r.region = deserialize_region(entry, "region");
    return r;
}

PointerAccessType PointerAccessMeta::ref() const {
    return std::unique_ptr<
        PointerAccessMeta,
        PtrAccessDeleter>(const_cast<PointerAccessMeta*>(this), PtrAccessDeleter(false));
}

PointerAccessType PointerAccessMeta::create_read_only(
    const symbolic::Expression& size, bool no_capture, std::optional<math::tensor::TensorLayout> layout
) {
    return std::unique_ptr<
        PointerAccessMeta,
        PtrAccessDeleter>(new PointerReadOnly(region_from_size(size, std::move(layout)), no_capture));
}

PointerAccessType PointerAccessMeta::create_invalidate() {
    return std::unique_ptr<PointerAccessMeta, PtrAccessDeleter>(new PointerInvalidate());
}

PointerAccessType PointerAccessMeta::create_full_write_only(
    const symbolic::Expression& size, bool no_capture, std::optional<math::tensor::TensorLayout> layout
) {
    return std::unique_ptr<PointerAccessMeta, PtrAccessDeleter>(
        new PointerFullWriteOnly(region_from_size(size, std::move(layout)), no_capture), PtrAccessDeleter(true)
    );
}

PointerAccessType PointerAccessMeta::create_generic(AccessRegion read, AccessRegion write, bool no_capture) {
    return std::unique_ptr<
        PointerAccessMeta,
        PtrAccessDeleter>(new PointerGenericAccess(std::move(read), std::move(write), no_capture));
}

PointerReadOnly::PointerReadOnly(std::optional<math::tensor::TensorLayout> region, bool no_capture)
    : region_(std::move(region)), no_capture_(no_capture) {
}

const math::tensor::TensorLayout* PointerReadOnly::read_layout() const {
    return region_ ? &*region_ : nullptr;
}

void PointerReadOnly::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (region_) {
        region_->replace_symbols(old_expression, new_expression);
    }
}

void PointerReadOnly::replace(const symbolic::ExpressionMapping& replacements) {
    if (region_) {
        region_->replace_symbols(replacements);
    }
}

PointerAccessType PointerReadOnly::clone() const {
    return PointerAccessType(new PointerReadOnly(region_, no_capture_));
}

void PointerReadOnly::serialize_to_json(nlohmann::json& entry) {
    entry["type"] = "PointerReadOnly";
    entry["no_capture"] = no_capture_;
    serialize_region(entry, "region", region_);
}

PointerFullWriteOnly::PointerFullWriteOnly(std::optional<math::tensor::TensorLayout> region, bool no_capture)
    : region_(std::move(region)), no_capture_(no_capture) {
}

const math::tensor::TensorLayout* PointerFullWriteOnly::write_layout() const {
    return region_ ? &*region_ : nullptr;
}

void PointerFullWriteOnly::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (region_) {
        region_->replace_symbols(old_expression, new_expression);
    }
}

void PointerFullWriteOnly::replace(const symbolic::ExpressionMapping& replacements) {
    if (region_) {
        region_->replace_symbols(replacements);
    }
}

PointerAccessType PointerFullWriteOnly::clone() const {
    return PointerAccessType(new PointerFullWriteOnly(region_, no_capture_));
}

void PointerFullWriteOnly::serialize_to_json(nlohmann::json& entry) {
    entry["type"] = "PointerWriteOnly";
    entry["no_capture"] = no_capture_;
    serialize_region(entry, "region", region_);
}

PointerGenericAccess::PointerGenericAccess(AccessRegion read, AccessRegion write, bool no_capture)
    : read_(std::move(read)), write_(std::move(write)), no_capture_(no_capture) {
}

const math::tensor::TensorLayout* PointerGenericAccess::read_layout() const {
    return read_.region ? &*read_.region : nullptr;
}

const math::tensor::TensorLayout* PointerGenericAccess::write_layout() const {
    return write_.region ? &*write_.region : nullptr;
}

void PointerGenericAccess::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (read_.region) {
        read_.region->replace_symbols(old_expression, new_expression);
    }
    if (write_.region) {
        write_.region->replace_symbols(old_expression, new_expression);
    }
}

void PointerGenericAccess::replace(const symbolic::ExpressionMapping& replacements) {
    if (read_.region) {
        read_.region->replace_symbols(replacements);
    }
    if (write_.region) {
        write_.region->replace_symbols(replacements);
    }
}

PointerAccessType PointerGenericAccess::clone() const {
    return PointerAccessType(new PointerGenericAccess(read_, write_, no_capture_));
}

void PointerGenericAccess::serialize_to_json(nlohmann::json& entry) {
    entry["type"] = "PointerGenericAccess";
    entry["no_capture"] = no_capture_;
    serialize_access_region(entry["read"], read_);
    serialize_access_region(entry["write"], write_);
}

PointerAccessType PointerInvalidate::clone() const {
    return PointerAccessType(new PointerInvalidate());
}

void PointerInvalidate::serialize_to_json(nlohmann::json& entry) {
    entry["type"] = "PointerInvalidate";
}

PointerAccessType PointerAccessMetaSerializer::deserialize(const nlohmann::json& entry) {
    if (entry.is_null()) {
        return nullptr;
    } else {
        auto type = entry.at("type").get<std::string>();

        if (type == "PointerInvalidate") {
            return PointerAccessMeta::create_invalidate();
        } else if (type == "PointerReadOnly") {
            return deserialize_read_only(entry);
        } else if (type == "PointerWriteOnly") {
            return deserialize_write_only(entry);
        } else if (type == "PointerGenericAccess") {
            return deserialize_generic(entry);
        } else {
            throw std::runtime_error("Unknown PointerAccessMeta type: " + type);
        }
    }
}

std::vector<PointerAccessType> PointerAccessMetaSerializer::deserialize_list(nlohmann::json::const_reference list) {
    std::vector<PointerAccessType> result;
    for (const auto& entry : list) {
        result.push_back(deserialize(entry));
    }
    return result;
}

std::vector<PointerAccessType> PointerAccessMetaSerializer::
    deserialize_list(nlohmann::json::const_iterator key, const nlohmann::json& parent) {
    if (key != parent.end()) {
        return deserialize_list(*key);
    } else {
        return {};
    }
}

PointerAccessType PointerAccessMetaSerializer::deserialize_read_only(nlohmann::json::const_reference entry) {
    bool no_capture = entry.at("no_capture").get<bool>();
    return PointerAccessType(new PointerReadOnly(deserialize_region(entry, "region"), no_capture));
}

PointerAccessType PointerAccessMetaSerializer::deserialize_write_only(nlohmann::json::const_reference entry) {
    bool no_capture = entry.at("no_capture").get<bool>();
    return PointerAccessType(new PointerFullWriteOnly(deserialize_region(entry, "region"), no_capture));
}

PointerAccessType PointerAccessMetaSerializer::deserialize_generic(nlohmann::json::const_reference entry) {
    bool no_capture = entry.at("no_capture").get<bool>();
    return PointerAccessMeta::create_generic(
        deserialize_access_region(entry.at("read")), deserialize_access_region(entry.at("write")), no_capture
    );
}

nlohmann::json PointerAccessMetaSerializer::serialize(const std::vector<PointerAccessType>& vector) {
    auto arr = nlohmann::json::array();
    for (const auto& entry : vector) {
        nlohmann::json j;
        entry->serialize_to_json(j);
        arr.push_back(j);
    }
    return arr;
}

} // namespace sdfg::data_flow
