#include "sdfg/transformations/transformation_schema.h"

#include "transformation_schema_json.h"

namespace sdfg {
namespace transformations {

const std::string& transformation_schema_text() {
    static const std::string schema(detail::TRANSFORMATION_SCHEMA_JSON);
    return schema;
}

const nlohmann::json& transformation_schema() {
    static const nlohmann::json schema = nlohmann::json::parse(transformation_schema_text());
    return schema;
}

} // namespace transformations
} // namespace sdfg
