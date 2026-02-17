#include "sdfg/passes/transformations/map_fusion_pass.h"

#include "sdfg/transformations/map_fusion.h"

namespace sdfg {
namespace passes {

MapFusion::MapFusion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool MapFusion::accept(structured_control_flow::Sequence& node) {
    bool applied = false;

    if (node.size() < 2) {
        return applied;
    }

    // Iterate over sequence looking for consecutive (Map, StructuredLoop) pairs
    size_t i = 0;
    while (i + 1 < node.size()) {
        auto* first = dynamic_cast<structured_control_flow::Map*>(&node.at(i).first);
        auto* second = dynamic_cast<structured_control_flow::StructuredLoop*>(&node.at(i + 1).first);

        if (first && second) {
            transformations::MapFusion transformation(*first, *second);
            if (transformation.can_be_applied(builder_, analysis_manager_)) {
                transformation.apply(builder_, analysis_manager_);
                applied = true;
            }
        }

        i++;
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
