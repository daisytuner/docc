#pragma once

#include "sdfg/passes/scheduler/sdfg_scheduler.h"

namespace sdfg {
namespace scheduling {

/// Leaves the SDFG as it is, for targets whose loop schedulers take it in any form.
class PassthroughScheduler : public passes::scheduler::SDFGScheduler {
public:
    bool schedule(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace scheduling
} // namespace sdfg
