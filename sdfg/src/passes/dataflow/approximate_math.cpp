#include "sdfg/passes/dataflow/approximate_math.h"

#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

namespace sdfg {
namespace passes {

ApproximateMath::ApproximateMath(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {
}

bool ApproximateMath::accept(structured_control_flow::Block& block) {
    bool applied = false;
    for (auto* lib_node : block.dataflow().library_nodes()) {
        auto* cmath = dynamic_cast<math::cmath::CMathNode*>(lib_node);
        if (cmath == nullptr || cmath->approximate()) {
            continue;
        }
        cmath->approximate(true);
        applied = true;
    }
    return applied;
}

} // namespace passes
} // namespace sdfg
