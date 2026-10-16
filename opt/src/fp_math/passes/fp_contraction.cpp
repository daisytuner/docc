#include "sdfg/fp_math/passes/fp_contraction.h"

#include "sdfg/fp_math/transformations/fuse_fma.h"

namespace sdfg {
namespace fp_math {

FPContraction::FPContraction(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {
}

bool FPContraction::accept(structured_control_flow::Block& block) {
    bool applied = false;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto* tasklet : block.dataflow().tasklets()) {
            FuseFMA transformation(*tasklet);
            if (transformation.try_apply(builder_, analysis_manager_)) {
                changed = true;
                applied = true;
                break;
            }
        }
    }
    return applied;
}

} // namespace fp_math
} // namespace sdfg
