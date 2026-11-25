#pragma once

#include "sdfg/passes/scheduler/sdfg_scheduler.h"

namespace sdfg {
namespace scheduling {

/**
 * @brief Brings the SDFG into docc's loop nest normal form before loop scheduling (see reordering::normalize), then
 * propagates symbols and removes the dead data and code this leaves.
 *
 * Normalizing loop nests a priori lets simple loop schedulers find the parallelism of complex applications.
 *
 * L. Trümper, P. Schaad, B. Ates, A. Calotoiu, M. Copik and T. Hoefler, "A Priori Loop Nest Normalization: Automatic
 * Loop Scheduling in Complex Applications", CGO 2025, https://doi.org/10.1145/3696443.3708951
 */
class APrioriLoopNestNormalization : public passes::scheduler::SDFGScheduler {
    bool enable_fusion_;

public:
    explicit APrioriLoopNestNormalization(bool enable_fusion = true);

    bool schedule(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace scheduling
} // namespace sdfg
