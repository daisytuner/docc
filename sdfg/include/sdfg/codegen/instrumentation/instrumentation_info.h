#pragma once

#include <cstddef>
#include <unordered_map>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/map.h"


namespace sdfg {
namespace codegen {


typedef StringEnum ElementType;
inline ElementType ElementType_Map{"map"};
inline ElementType ElementType_Reduce{"reduce"};
inline ElementType ElementType_For{"for"};
inline ElementType ElementType_While{"while"};
inline ElementType ElementType_Block{"block"};
inline ElementType ElementType_IfElse{"if_else"};
inline ElementType ElementType_Sequence{"sequence"};
inline ElementType ElementType_H2DTransfer{"h2d_transfer"};
inline ElementType ElementType_D2HTransfer{"d2h_transfer"};
inline ElementType ElementType_Math{"math"};
inline ElementType ElementType_Unknown{"unknown"};

typedef StringEnum TargetType;
inline TargetType TargetType_SEQUENTIAL{structured_control_flow::ScheduleType_Sequential::value()};
// Legacy name for OMP parallelism
inline TargetType TargetType_CPU_PARALLEL{"CPU_PARALLEL"};

class InstrumentationInfo {
private:
    // General properties
    size_t element_id_;
    ElementType element_type_;
    TargetType target_type_;
    analysis::LoopInfo loop_info_;

    std::unordered_map<std::string, std::string> metrics_;

public:
    InstrumentationInfo(
        size_t element_id,
        const ElementType& element_type,
        const TargetType& target_type,
        const analysis::LoopInfo& loop_info = {},
        const std::unordered_map<std::string, std::string>& metrics = {}
    );

    size_t element_id() const;

    const ElementType& element_type() const;

    const TargetType& target_type() const;

    const analysis::LoopInfo& loop_info() const;

    const std::unordered_map<std::string, std::string>& metrics() const;
};

} // namespace codegen
} // namespace sdfg
