#include "sdfg/passes/normalization/map_fusion.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/transformations/map_fusion.h"

namespace sdfg {
namespace passes {
namespace normalization {

MapFusion::MapFusion(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool allow_init_hoist
)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager), allow_init_hoist_(allow_init_hoist) {}

bool MapFusion::accept(structured_control_flow::Sequence& node) {
    bool applied = false;

    if (node.size() < 2) {
        return applied;
    }

    // LoopAnalysis is cheap to re-run and must be re-run whenever the loop structure changes.
    // AssumptionsAnalysis is expensive (symbolic computation) and is preserved across
    // ProducerIntoConsumer fusions: those only prepend Block nodes to the consumer body, so
    // the existing per-node assumption entries remain valid for all nodes that are not new.
    //
    // stale_loop tracks the StructuredLoop whose body was just modified by a
    // ProducerIntoConsumer fusion.  When it is encountered as the next producer candidate,
    // LoopAnalysis is invalidated so it is re-run with the updated body.
    structured_control_flow::StructuredLoop* stale_loop = nullptr;

    // Iterate over sequence looking for consecutive (Map, StructuredLoop) pairs
    size_t i = 0;
    while (i + 1 < node.size()) {
        auto* first = dyn_cast<structured_control_flow::Map*>(&node.at(i).first);
        if (!first) {
            stale_loop = nullptr;
            i++;
            continue;
        }
        if (first->root().size() == 0) {
            i++;
            continue;
        }

        // If this node was the consumer of a previous ProducerIntoConsumer fusion its
        // LoopAnalysis entry is stale (blocks were prepended to its body).  Force a
        // re-run before checking it as a new producer.
        if (stale_loop == first) {
            analysis_manager_.invalidate<analysis::LoopAnalysis>();
            stale_loop = nullptr;
        }

        if (auto* second = dyn_cast<structured_control_flow::StructuredLoop*>(&node.at(i + 1).first)) {
            if (second->root().size() == 0) {
                i++;
                continue;
            }
            transformations::MapFusion transformation(*first, *second, true, allow_init_hoist_);
            if (transformation.can_be_applied(builder_, analysis_manager_)) {
                auto first_name = first->indvar()->get_name();
                auto second_name = second->indvar()->get_name();

                // Apply without letting the transformation invalidate the manager;
                // we selectively invalidate below based on the fusion direction.
                transformation.apply_without_invalidate(builder_, analysis_manager_);

                if (transformation.was_producer_into_consumer()) {
                    // ProducerIntoConsumer only prepends Block nodes to second's body.
                    // The loop structure of every other node in this sequence is unchanged,
                    // so AssumptionsAnalysis and ArgumentsAnalysis stay valid.
                    // LoopAnalysis for `second` is now stale and must be re-run when
                    // `second` is next encountered as a producer candidate.
                    analysis_manager_.invalidate<analysis::LoopAnalysis>();
                    stale_loop = second;
                } else {
                    // ConsumerIntoProducer removes second_loop_ from the sequence and
                    // modifies the producer body — full invalidation is required.
                    analysis_manager_.invalidate_all();
                    stale_loop = nullptr;
                }

                DEBUG_PRINTLN("Applied MapFusion to maps " + first_name + " and " + second_name);
                applied = true;
            }
        } else if (i + 2 < node.size()) {
            auto* mid_block = dyn_cast<structured_control_flow::Block*>(&node.at(i + 1).first);
            if (mid_block && mid_block->is_a_library_node<stdlib::MallocNode>()) {
                if (auto* second = dyn_cast<structured_control_flow::StructuredLoop*>(&node.at(i + 2).first)) {
                    if (second->root().size() == 0) {
                        i++;
                        continue;
                    }
                    transformations::MapFusion transformation(*first, *second, false, allow_init_hoist_);
                    if (transformation.can_be_applied(builder_, analysis_manager_)) {
                        auto first_name = first->indvar()->get_name();
                        auto second_name = second->indvar()->get_name();

                        transformation.apply_without_invalidate(builder_, analysis_manager_);

                        if (transformation.was_producer_into_consumer()) {
                            analysis_manager_.invalidate<analysis::LoopAnalysis>();
                            stale_loop = second;
                        } else {
                            analysis_manager_.invalidate_all();
                            stale_loop = nullptr;
                        }

                        DEBUG_PRINTLN(
                            "Applied MapFusion to map " + first_name + " and loop " + second_name +
                            " with intermediate malloc block"
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
