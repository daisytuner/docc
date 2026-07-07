#include "sdfg/codegen/instrumentation/instrumentation_info.h"


namespace sdfg {
namespace codegen {


InstrumentationInfo::InstrumentationInfo(
    size_t element_id,
    std::string_view element_desc,
    const TargetType& target_type,
    const analysis::LoopInfo& loop_info,
    const std::unordered_map<std::string, std::string>& metrics
)
    : element_id_(element_id), element_desc_(element_desc), target_type_(target_type), loop_info_(loop_info),
      metrics_(metrics) {}

size_t InstrumentationInfo::element_id() const { return element_id_; }

const std::string& InstrumentationInfo::element_desc() const { return element_desc_; }

const TargetType& InstrumentationInfo::target_type() const { return target_type_; }

const analysis::LoopInfo& InstrumentationInfo::loop_info() const { return loop_info_; }

const std::unordered_map<std::string, std::string>& InstrumentationInfo::metrics() const { return metrics_; }

} // namespace codegen
} // namespace sdfg
