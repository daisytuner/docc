#include "sdfg/passes/normalization/perfect_loop_distribution.h"

#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>

namespace sdfg {
namespace passes {
namespace normalization {

bool PerfectLoopDistribution::can_be_applied(structured_control_flow::StructuredLoop& loop) {
    if (loop.root().size() == 1) {
        return false;
    }

    bool has_subloop = false;
    for (size_t i = 0; i < loop.root().size(); i++) {
        // skip blocks
        if (dynamic_cast<structured_control_flow::Block*>(&loop.root().at(i).first)) {
            continue;
        }
        if (dynamic_cast<structured_control_flow::StructuredLoop*>(&loop.root().at(i).first)) {
            has_subloop = true;
            break;
        }
        // if not a block or a loop, then we can't apply the transformation
        return false;
    }
    if (!has_subloop) {
        return false;
    }

    transformations::LoopDistribute loop_distribute(loop);
    if (!loop_distribute.can_be_applied(builder_, analysis_manager_)) {
        return false;
    }

    return true;
};

void PerfectLoopDistribution::apply(structured_control_flow::StructuredLoop& loop) {
    transformations::LoopDistribute loop_distribute(loop);
    loop_distribute.apply(builder_, analysis_manager_);
};

PerfectLoopDistribution::
    PerfectLoopDistribution(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {

      };

bool PerfectLoopDistribution::accept(structured_control_flow::For& node) {
    if (can_be_applied(node)) {
        apply(node);
        return true;
    }
    return false;
};

bool PerfectLoopDistribution::accept(structured_control_flow::Map& node) {
    if (can_be_applied(node)) {
        apply(node);
        return true;
    }
    return false;
};

} // namespace normalization
} // namespace passes
} // namespace sdfg
