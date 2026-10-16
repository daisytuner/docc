#include "sdfg/fp_math/passes/fp_contraction.h"

#include <vector>

#include "sdfg/fp_math/transformations/fuse_fma.h"

namespace sdfg {
namespace fp_math {

FPContraction::FPContraction(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {
}

bool FPContraction::visit() {
    bool applied = visitor::NonStoppingStructuredSDFGVisitor::visit();
    if (applied) {
        analysis_manager_.invalidate_all();
    }
    return applied;
}

bool FPContraction::accept(structured_control_flow::Block& block) {
    bool applied = false;
    auto tasklets = block.dataflow().tasklets();
    std::vector<data_flow::Tasklet*> additions;
    for (auto* tasklet : tasklets) {
        if (tasklet->code() == data_flow::TaskletCode::fp_add) {
            additions.push_back(tasklet);
        }
    }
    for (auto* addition : additions) {
        FuseFMA transformation(*addition, false);
        if (transformation.try_apply(builder_, analysis_manager_)) {
            applied = true;
        }
    }
    return applied;
}

} // namespace fp_math
} // namespace sdfg
