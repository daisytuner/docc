#include "sdfg/passes/normalization/map_fusion.h"

#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/transformations/map_fusion.h"

namespace sdfg {
namespace passes {
namespace normalization {

MapFusion::MapFusion(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    bool allow_init_hoist,
    bool allow_prod_into_cons
)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager), allow_init_hoist_(allow_init_hoist),
      allow_prod_into_cons_(allow_prod_into_cons) {}

bool MapFusion::accept(structured_control_flow::Sequence& node) {
    bool applied = false;

    if (node.size() < 2) {
        return applied;
    }

    // Iterate over sequence looking for consecutive (Map, StructuredLoop) pairs
    size_t i = 0;
    while (i + 1 < node.size()) {
        auto* first = dyn_cast<structured_control_flow::Map*>(&node.at(i));
        if (!first) {
            i++;
            continue;
        }
        if (first->root().size() == 0) {
            i++;
            continue;
        }

        if (auto* second = dyn_cast<structured_control_flow::StructuredLoop*>(&node.at(i + 1))) {
            if (second->root().size() == 0) {
                i++;
                continue;
            }
            transformations::MapFusion transformation(*first, *second, true, allow_init_hoist_, allow_prod_into_cons_);
            if (transformation.can_be_applied(builder_, analysis_manager_)) {
                auto first_id = first->element_id();
                auto second_id = second->element_id();
                transformation.apply(builder_, analysis_manager_);
                DEBUG_PRINTLN(
                    "Applied MapFusion to #" + std::to_string(first_id) + " " +
                    (transformation.last_fusion_direction() ==
                             loop_fusion::LoopFusionByAccessWorker::FusionDirection::ProducerIntoConsumer
                         ? "->"
                         : "<-") +
                    " #" + std::to_string(second_id)
                );
                applied = true;
            }
        } else if (i + 2 < node.size()) {
            auto* mid_block = dyn_cast<structured_control_flow::Block*>(&node.at(i + 1));
            if (mid_block && mid_block->is_a_library_node<stdlib::MallocNode>()) {
                if (auto* second = dyn_cast<structured_control_flow::StructuredLoop*>(&node.at(i + 2))) {
                    if (second->root().size() == 0) {
                        i++;
                        continue;
                    }
                    transformations::MapFusion
                        transformation(*first, *second, false, allow_init_hoist_, allow_prod_into_cons_);
                    if (transformation.can_be_applied(builder_, analysis_manager_)) {
                        auto first_id = first->element_id();
                        auto second_id = second->element_id();
                        transformation.apply(builder_, analysis_manager_);
                        DEBUG_PRINTLN(
                            "Applied MapFusion to #" + std::to_string(first_id) + " " +
                            (transformation.last_fusion_direction() ==
                                     loop_fusion::LoopFusionByAccessWorker::FusionDirection::ProducerIntoConsumer
                                 ? "->"
                                 : "<-") +
                            " #" + std::to_string(second_id) + " with intermediate malloc block"
                        );
                        applied = true;

                        // Move malloc block before the first map
                        this->builder_.move_child(node, i + 1, node, i);
                        i = i + 2; // Skip over the newly moved malloc block and the second loop that was just fused
                        continue;
                    }
                }
            }
        }
        i++;
    }

    return applied;
}

} // namespace normalization
} // namespace passes
} // namespace sdfg
