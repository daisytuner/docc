#include "sdfg/data_flow/pointer_metadata.h"

#include "cereal/types/utility.hpp"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg::data_flow {

MemoryAccessPatternType MemoryAccessPattern::ref() const {
    return std::unique_ptr<
        MemoryAccessPattern,
        AccessPatternsDeleter>(const_cast<MemoryAccessPattern*>(this), AccessPatternsDeleter(false));
}

MemoryAccessPatternType ConvexAccessPattern::clone() const {
    return std::unique_ptr<MemoryAccessPattern, AccessPatternsDeleter>(new ConvexAccessPattern(size_));
}

void ConvexAccessPattern::serialize_to_json(nlohmann::json& entry) {
    entry["type"] = "ConvexAccessPattern";
    serializer::JSONSerializer serializer;
    entry["size"] = serializer.expression(size_);
    entry["not_sparse"] = not_sparse_;
}

MemoryAccessPatternType ConvexAccessPattern::create(symbolic::Expression size, bool not_sparse) {
    return std::unique_ptr<MemoryAccessPattern, AccessPatternsDeleter>(new ConvexAccessPattern(size, not_sparse));
}

MemoryAccessPatternType NoAccessPattern::instance() {
    static std::unique_ptr<MemoryAccessPattern, AccessPatternsDeleter> instance(new NoAccessPattern());
    return instance->ref();
}

void NoAccessPattern::serialize_to_json(nlohmann::json& entry) { entry["type"] = "NoAccessPattern"; }

MemoryAccessPatternType NoAccessPattern::clone() const { return this->ref(); }

PointerAccessType PointerAccessMeta::ref() const {
    return std::unique_ptr<
        PointerAccessMeta,
        PtrAccessDeleter>(const_cast<PointerAccessMeta*>(this), PtrAccessDeleter(false));
}

PointerAccessType PointerAccessMeta::create_read_only(const symbolic::Expression& size, bool no_capture) {
    return std::unique_ptr<PointerAccessMeta, PtrAccessDeleter>(new PointerReadOnly(size, no_capture));
}

PointerAccessType PointerAccessMeta::create_invalidate() {
    return std::unique_ptr<PointerAccessMeta, PtrAccessDeleter>(new PointerInvalidate());
}

PointerAccessType PointerAccessMeta::create_full_write_only(const symbolic::Expression& size, bool no_capture) {
    return std::unique_ptr<
        PointerAccessMeta,
        PtrAccessDeleter>(new PointerFullWriteOnly(size, no_capture), PtrAccessDeleter(true));
}

PointerAccessType PointerAccessMeta::
    create_generic(MemoryAccessPatternType read_pattern, MemoryAccessPatternType write_pattern, bool no_capture) {
    return std::unique_ptr<
        PointerAccessMeta,
        PtrAccessDeleter>(new PointerGenericAccess(std::move(read_pattern), std::move(write_pattern), no_capture));
}

PointerReadOnly::PointerReadOnly(symbolic::Expression size, bool no_capture) : size_(size), no_capture_(no_capture) {}

MemoryAccessPatternType PointerReadOnly::access_read_pattern() const {
    if (size_.is_null()) {
        return {};
    } else {
        return ConvexAccessPattern::create(size_);
    }
}

MemoryAccessPatternType PointerReadOnly::access_write_pattern() const { return NoAccessPattern::instance(); }

void PointerReadOnly::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    size_ = symbolic::subs(size_, old_expression, new_expression);
}

void PointerReadOnly::replace(const symbolic::ExpressionMapping& replacements) {
    size_ = SymEngine::subs(size_, replacements);
}

PointerAccessType PointerReadOnly::clone() const { return PointerAccessType(new PointerReadOnly(size_, no_capture_)); }

void PointerReadOnly::serialize_to_json(nlohmann::json& entry) {
    serializer::JSONSerializer serializer;
    entry["type"] = "PointerReadOnly";
    entry["size"] = serializer.expression(size_);
    entry["no_capture"] = no_capture_;
}

PointerFullWriteOnly::PointerFullWriteOnly(symbolic::Expression size, bool no_capture)
    : size_(size), no_capture_(no_capture) {}

MemoryAccessPatternType PointerFullWriteOnly::access_write_pattern() const {
    if (size_.is_null()) {
        return nullptr;
    } else {
        return ConvexAccessPattern::create(size_, true);
    }
}

MemoryAccessPatternType PointerFullWriteOnly::access_read_pattern() const { return NoAccessPattern::instance(); }

void PointerFullWriteOnly::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    size_ = symbolic::subs(size_, old_expression, new_expression);
}

void PointerFullWriteOnly::replace(const symbolic::ExpressionMapping& replacements) {
    size_ = SymEngine::subs(size_, replacements);
}

PointerAccessType PointerFullWriteOnly::clone() const {
    return PointerAccessType(new PointerFullWriteOnly(size_, no_capture_));
}

void PointerFullWriteOnly::serialize_to_json(nlohmann::json& entry) {
    serializer::JSONSerializer serializer;
    entry["type"] = "PointerWriteOnly";
    entry["size"] = serializer.expression(size_);
    entry["no_capture"] = no_capture_;
}

PointerGenericAccess::
    PointerGenericAccess(MemoryAccessPatternType read_pattern, MemoryAccessPatternType write_pattern, bool no_capture)
    : read_pattern_(std::move(read_pattern)), write_pattern_(std::move(write_pattern)), no_capture_(no_capture) {}

MemoryAccessPatternType PointerGenericAccess::access_read_pattern() const {
    return read_pattern_ ? read_pattern_->ref() : nullptr;
}

MemoryAccessPatternType PointerGenericAccess::access_write_pattern() const {
    return write_pattern_ ? write_pattern_->ref() : nullptr;
}

bool PointerGenericAccess::may_contain_reads() const { return !read_pattern_ || !read_pattern_->empty(); }

bool PointerGenericAccess::may_contain_writes() const { return !write_pattern_ || !write_pattern_->empty(); }

void PointerGenericAccess::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (read_pattern_) {
        read_pattern_->replace(old_expression, new_expression);
    }
    if (write_pattern_) {
        write_pattern_->replace(old_expression, new_expression);
    }
}

void PointerGenericAccess::replace(const symbolic::ExpressionMapping& replacements) {
    if (read_pattern_) {
        read_pattern_->replace(replacements);
    }
    if (write_pattern_) {
        write_pattern_->replace(replacements);
    }
}

PointerAccessType PointerGenericAccess::clone() const {
    auto rd_patt_new = read_pattern_ ? read_pattern_->clone() : nullptr;
    auto wr_patt_new = write_pattern_ ? write_pattern_->clone() : nullptr;
    return PointerAccessType(new PointerGenericAccess(std::move(rd_patt_new), std::move(wr_patt_new), no_capture_));
}

void PointerGenericAccess::serialize_to_json(nlohmann::json& entry) {
    entry["type"] = "PointerGenericAccess";
    if (read_pattern_) {
        read_pattern_->serialize_to_json(entry["read_pattern"]);
    }
    if (write_pattern_) {
        write_pattern_->serialize_to_json(entry["write_pattern"]);
    }
    entry["no_capture"] = no_capture_;
}

PointerAccessType PointerInvalidate::clone() const { return PointerAccessType(new PointerInvalidate()); }

void PointerInvalidate::serialize_to_json(nlohmann::json& entry) { entry["type"] = "PointerInvalidate"; }

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
            throw std::runtime_error("Unknown MemoryAccessPattern type: " + type);
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
    serializer::JSONSerializer serializer;
    bool no_capture = entry.at("no_capture").get<bool>();
    auto size = serializer.json_to_expr(entry.at("size"));
    return PointerAccessMeta::create_read_only(size, no_capture);
}

PointerAccessType PointerAccessMetaSerializer::deserialize_write_only(nlohmann::json::const_reference entry) {
    serializer::JSONSerializer serializer;
    bool no_capture = entry.at("no_capture").get<bool>();
    auto size = serializer.json_to_expr(entry.at("size"));
    return PointerAccessMeta::create_full_write_only(size, no_capture);
}

PointerAccessType PointerAccessMetaSerializer::deserialize_generic(nlohmann::json::const_reference entry) {
    auto read_pattern = deserialize_access_pattern(entry.at("read_pattern"));
    auto write_pattern = deserialize_access_pattern(entry.at("write_pattern"));
    bool no_capture = entry.at("no_capture").get<bool>();
    return PointerAccessMeta::create_generic(std::move(read_pattern), std::move(write_pattern), no_capture);
}

MemoryAccessPatternType PointerAccessMetaSerializer::deserialize_convex_pattern(nlohmann::json::const_reference entry) {
    serializer::JSONSerializer serializer;
    auto size = serializer.json_to_expr(entry.at("size"));
    bool not_sparse = entry.at("not_sparse").get<bool>();
    return ConvexAccessPattern::create(size, not_sparse);
}

MemoryAccessPatternType PointerAccessMetaSerializer::deserialize_access_pattern(nlohmann::json::const_reference entry) {
    if (entry.is_null()) {
        return nullptr;
    } else {
        auto type = entry.at("type").get<std::string>();
        if (type == "NoAccessPattern") {
            return NoAccessPattern::instance();
        } else if (type == "ConvexAccessPattern") {
            return deserialize_convex_pattern(entry);
        } else {
            throw std::runtime_error("Unknown MemoryAccessPattern type: " + type);
        }
    }
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
