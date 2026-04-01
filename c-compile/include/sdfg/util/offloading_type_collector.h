#pragma once

#include <sdfg/data_flow/library_nodes/math/blas/blas_node.h>
#include <sdfg/data_flow/library_nodes/math/math_node.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>

namespace sdfg::auto_util {

/**
 * Filters over elements that have a scheduleType or implementationType like Tenstorrent, CUDA etc.
 * So that we can detect whether we are currently making device-specific references (which require additional runtimes)
 * in a convenient way. Because for the choice which runtimes and compile features we need to enable, we do not
 * understand whether it was transformed map or a library node that created the dependency. No particular reason for
 * ScheduleType being the result. It just existed first. Ideally it'd be a custom type represents the runtime
 * dependency. But will be outdated, once we support multiple different offloading runtimes simultaneously.
 *
 * lives in sdfg-lib auto, because the types it needs to recognize are here. Allows changing sdfglib-auto more without
 * also requiring changes in DOCC
 */
class OffloadingTypeCollector : public sdfg::visitor::StructuredSDFGVisitor {
private:
    ScheduleType schedule_type_;

public:
    OffloadingTypeCollector(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        const ScheduleType& sched_type
    );

    bool accept(sdfg::structured_control_flow::Block& node) override;

    bool accept(sdfg::structured_control_flow::Map& map) override;

    const ScheduleType& schedule_type() const;

private:
    void found_schedule_type(const ScheduleType& sched_type);
};

} // namespace sdfg::auto_util
