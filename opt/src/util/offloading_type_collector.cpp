#include "sdfg/util/offloading_type_collector.h"
#include "sdfg/data_flow/library_nodes/math/math.h"

namespace sdfg::auto_util {

OffloadingTypeCollector::OffloadingTypeCollector(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, const ScheduleType& sched_type
)
    : StructuredSDFGVisitor(builder, analysis_manager), schedule_type_(sched_type) {}

void OffloadingTypeCollector::found_schedule_type(const ScheduleType& sched_type) {
    if (sched_type.value() != ScheduleType_Sequential::value()) { // not the base type
        // TT uses multiple, but for this check we can treat them as one!
        auto mapped_sched_type = sdfg::tenstorrent::is_tenstorrent_schedule(sched_type)
                                     ? sdfg::tenstorrent::ScheduleType_Tenstorrent_Kernel::create()
                                     : sched_type;
        if (sched_type.value() != ScheduleType_Sequential::value()) { // first special type, expect all further ones
            // to be base or the same
            schedule_type_ = mapped_sched_type;
        } else if (schedule_type_.value() != mapped_sched_type.value()) { // there can only be 1 special type per module
                                                                          // for now
            throw std::runtime_error(
                "Mixed schedule types in ConditionalSDFG: " + schedule_type_.value() + " and " +
                mapped_sched_type.value()
            );
        }
    }
}

const ScheduleType& OffloadingTypeCollector::schedule_type() const { return schedule_type_; }

bool OffloadingTypeCollector::accept(Map& map) {
    auto& map_schedule_type = map.schedule_type();
    found_schedule_type(map_schedule_type);
    return false;
}

bool OffloadingTypeCollector::accept(Block& node) {
    auto& dataflow = node.dataflow();
    for (auto& library_node : dataflow.nodes()) {
        if (auto math_node = dynamic_cast<sdfg::math::MathNode*>(&library_node)) {
            auto& impl_type = math_node->implementation_type();
            if (impl_type == sdfg::tenstorrent::ImplementationType_Tenstorrent_WithTransfers ||
                impl_type == sdfg::tenstorrent::ImplementationType_Tenstorrent_WithoutTransfers) {
                sdfg::structured_control_flow::ScheduleType schedule_type =
                    sdfg::tenstorrent::ScheduleType_Tenstorrent_Kernel::create();
                found_schedule_type(schedule_type);
            } else if (impl_type == sdfg::cuda::blas::ImplementationType_CUBLASWithTransfers ||
                       impl_type == sdfg::cuda::blas::ImplementationType_CUBLASWithoutTransfers) {
                sdfg::structured_control_flow::ScheduleType schedule_type = sdfg::cuda::ScheduleType_CUDA::create();
                found_schedule_type(schedule_type);
            } else if (impl_type == sdfg::math::blas::ImplementationType_BLAS ||
                       impl_type == sdfg::data_flow::ImplementationType_NONE) {
                sdfg::structured_control_flow::ScheduleType schedule_type =
                    sdfg::structured_control_flow::ScheduleType_Sequential::create();
                found_schedule_type(schedule_type);
            } else {
                throw std::runtime_error("Unsupported math node implementation type: " + impl_type.value());
            }
        }
    }

    return false;
}

} // namespace sdfg::auto_util
