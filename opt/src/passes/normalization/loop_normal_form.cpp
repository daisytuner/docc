#include "sdfg/passes/normalization/loop_normal_form.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/loop_condition_normalize.h"
#include "sdfg/transformations/loop_indvar_finalize.h"
#include "sdfg/transformations/loop_rotate.h"
#include "sdfg/transformations/loop_shift.h"
#include "sdfg/transformations/loop_unit_stride.h"

namespace sdfg {
namespace passes {
namespace normalization {

LoopNormalFormPass::LoopNormalFormPass() {};

bool LoopNormalFormPass::apply(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop
) {
    bool applied = false;

    // Step 1: Shift loop to start from 0
    transformations::LoopShift loop_shift(loop);
    if (loop_shift.can_be_applied(builder, analysis_manager)) {
        loop_shift.apply(builder, analysis_manager);
        applied = true;
    }

    // Step 2: Convert non-unit stride to unit stride
    transformations::LoopUnitStride loop_unit_stride(loop);
    if (loop_unit_stride.can_be_applied(builder, analysis_manager)) {
        loop_unit_stride.apply(builder, analysis_manager);
        applied = true;
    }

    // Step 3: Convert != conditions to < or > (requires unit stride)
    transformations::LoopConditionNormalize loop_cond_normalize(loop);
    if (loop_cond_normalize.can_be_applied(builder, analysis_manager)) {
        loop_cond_normalize.apply(builder, analysis_manager);
        applied = true;
    }

    // Step 4: Convert negative stride to positive (requires < or > conditions)
    transformations::LoopRotate loop_rotate(loop);
    if (loop_rotate.can_be_applied(builder, analysis_manager)) {
        loop_rotate.apply(builder, analysis_manager);
        applied = true;

        // After rotation, we may have a non-zero init, so try shifting
        transformations::LoopShift loop_shift(loop);
        if (loop_shift.can_be_applied(builder, analysis_manager)) {
            loop_shift.apply(builder, analysis_manager);
            applied = true;
        }

        transformations::LoopConditionNormalize loop_cond_normalize(loop);
        if (loop_cond_normalize.can_be_applied(builder, analysis_manager)) {
            loop_cond_normalize.apply(builder, analysis_manager);
            applied = true;
        }
    }

    // Step 5: Simplify reconstruction blocks with closed-form expression
    transformations::LoopIndvarFinalize finalize(loop);
    if (finalize.can_be_applied(builder, analysis_manager)) {
        finalize.apply(builder, analysis_manager);
        applied = true;
    }

    return applied;
};

bool LoopNormalFormPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool modified = false;

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    for (auto& loop : loop_analysis.loops()) {
        if (!dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            continue; // Skip non-structured loops
        }
        modified |= apply(builder, analysis_manager, *dynamic_cast<structured_control_flow::StructuredLoop*>(loop));
    }

    return modified;
}


} // namespace normalization
} // namespace passes
} // namespace sdfg
