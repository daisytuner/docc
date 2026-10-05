#include "sdfg/codegen/instrumentation/instrumentation_info.h"


namespace sdfg {
namespace codegen {


InstrumentationInfo::InstrumentationInfo(
    size_t element_id,
    std::string_view element_desc,
    const TargetType& target_type,
    InstrumentationEventType event_type,
    const analysis::LoopInfo& loop_info,
    const std::unordered_map<std::string, std::string>& metrics
)
    : element_id_(element_id), element_desc_(element_desc), target_type_(target_type), event_type_(event_type),
      loop_info_(loop_info), metrics_(metrics) {
}

size_t InstrumentationInfo::element_id() const {
    return element_id_;
}

const std::string& InstrumentationInfo::element_desc() const {
    return element_desc_;
}

const TargetType& InstrumentationInfo::target_type() const {
    return target_type_;
}

InstrumentationEventType InstrumentationInfo::event_type() const {
    return event_type_;
}

const analysis::LoopInfo& InstrumentationInfo::loop_info() const {
    return loop_info_;
}

const std::unordered_map<std::string, std::string>& InstrumentationInfo::metrics() const {
    return metrics_;
}

bool InstrumentationInfo::sampling() const {
    return sampling_;
}

void InstrumentationInfo::set_sampling(bool sampling) {
    sampling_ = sampling;
}

std::optional<ElementId> InstrumentationInfo::logical_region_id() const {
    return logical_region_id_;
}

void InstrumentationInfo::set_logical_region_id(std::optional<ElementId> logical_region_id) {
    logical_region_id_ = logical_region_id;
}

std::optional<ElementId> InstrumentationInfo::source_loop_id() const {
    return source_loop_id_;
}

void InstrumentationInfo::set_source_loop_id(std::optional<ElementId> source_loop_id) {
    source_loop_id_ = source_loop_id;
}

const std::optional<nlohmann::json>& InstrumentationInfo::expected_performance() const {
    return expected_performance_;
}

void InstrumentationInfo::set_expected_performance(std::optional<nlohmann::json> expected_performance) {
    expected_performance_ = std::move(expected_performance);
}

std::optional<double> InstrumentationInfo::vector_distance() const {
    return vector_distance_;
}

void InstrumentationInfo::set_vector_distance(std::optional<double> vector_distance) {
    vector_distance_ = vector_distance;
}

const std::vector<InstrumentationMemberInfo>& InstrumentationInfo::members() const {
    return members_;
}

void InstrumentationInfo::set_members(std::vector<InstrumentationMemberInfo> members) {
    members_ = std::move(members);
}

} // namespace codegen
} // namespace sdfg
