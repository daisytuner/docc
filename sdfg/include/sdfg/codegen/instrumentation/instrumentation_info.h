#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/map.h"


namespace sdfg {
namespace codegen {

enum InstrumentationEventType { CPU = 0, CUDA = 1, NONE = 2 };

struct InstrumentationMemberInfo {
    ElementId element_id;
    std::string filename;
    std::string function;
    size_t start_line;
    size_t start_column;
    size_t end_line;
    size_t end_column;
};

typedef StringEnum TargetType;
inline TargetType TargetType_SEQUENTIAL{structured_control_flow::ScheduleType_Sequential::value()};
// Legacy name for OMP parallelism
inline TargetType TargetType_CPU_PARALLEL{"CPU_PARALLEL"};

class InstrumentationInfo {
private:
    // General properties
    size_t element_id_;
    std::string element_desc_;
    TargetType target_type_;
    InstrumentationEventType event_type_;
    analysis::LoopInfo loop_info_;
    std::unordered_map<std::string, std::string> metrics_;
    std::optional<ElementId> logical_region_id_;
    std::optional<ElementId> original_loop_id_;
    std::optional<double> expected_speedup_;
    std::optional<double> vector_distance_;
    std::vector<InstrumentationMemberInfo> members_;
    bool sampling_ = false;

public:
    InstrumentationInfo(
        size_t element_id,
        std::string_view element_desc,
        const TargetType& target_type,
        InstrumentationEventType event_type,
        const analysis::LoopInfo& loop_info = {},
        const std::unordered_map<std::string, std::string>& metrics = {}
    );

    size_t element_id() const;

    const std::string& element_desc() const;

    const TargetType& target_type() const;

    InstrumentationEventType event_type() const;

    const analysis::LoopInfo& loop_info() const;

    const std::unordered_map<std::string, std::string>& metrics() const;

    bool sampling() const;

    void set_sampling(bool sampling);

    std::optional<ElementId> logical_region_id() const;

    void set_logical_region_id(std::optional<ElementId> logical_region_id);

    std::optional<ElementId> original_loop_id() const;

    void set_original_loop_id(std::optional<ElementId> original_loop_id);

    std::optional<double> expected_speedup() const;

    void set_expected_speedup(std::optional<double> expected_speedup);

    std::optional<double> vector_distance() const;

    void set_vector_distance(std::optional<double> vector_distance);

    const std::vector<InstrumentationMemberInfo>& members() const;

    void set_members(std::vector<InstrumentationMemberInfo> members);
};

} // namespace codegen
} // namespace sdfg
