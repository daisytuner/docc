#pragma once

#include <string>

#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/structured_control_flow/map.h>

namespace sdfg {
namespace vectorize {

/**
 * @brief Vectorize schedule type
 *
 * Indicates that loop iterations can be vectorized using pragmas.
 */
class ScheduleType_Vectorize {
public:
    static const std::string value() { return "VECTORIZE"; }

    static structured_control_flow::ScheduleType create() {
        return structured_control_flow::ScheduleType(value(), structured_control_flow::ScheduleTypeCategory::Vectorizer);
    }
};

} // namespace vectorize
} // namespace sdfg
