#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace passes {
namespace scheduler {

/**
 * @brief Brings a whole SDFG into the normal form a target's loop schedulers expect.
 *
 * Each target creates one (see docc::target::TargetScheduling). It runs before remote tuning and the
 * target's loop schedulers, and must be deterministic: the tuning server runs it as well.
 */
class SDFGScheduler {
public:
    virtual ~SDFGScheduler() = default;

    /// @return true if the SDFG was modified
    virtual bool schedule(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) = 0;
};

} // namespace scheduler
} // namespace passes
} // namespace sdfg
