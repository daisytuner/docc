#include "sdfg/passes/normalization/map_fusion.h"

#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/transformations/map_fusion.h"

namespace sdfg {
namespace passes {
namespace normalization {

std::string is_malloc_block(const structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    if (dataflow.nodes().size() != 2 || dataflow.edges().size() != 1) {
        return "";
    }
    auto lib_nodes = dataflow.library_nodes();
    if (lib_nodes.size() != 1) {
        return "";
    }
    auto* libnode = *lib_nodes.begin();
    if (!dynamic_cast<const stdlib::MallocNode*>(libnode)) {
        return "";
    }

    auto& oedge = *dataflow.out_edges(*libnode).begin();
    auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
    return dst.data();
}

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
        if (!first) {
            i++;
            continue;
        }
        if (first->root().size() == 0) {
            i++;
            continue;
        }

        if (auto* second = dynamic_cast<structured_control_flow::StructuredLoop*>(&node.at(i + 1).first)) {
            if (second->root().size() == 0) {
                i++;
                continue;
            }
            transformations::MapFusion transformation(*first, *second);
            if (transformation.can_be_applied(builder_, analysis_manager_)) {
                auto first_name = first->indvar()->get_name();
                auto second_name = second->indvar()->get_name();
                transformation.apply(builder_, analysis_manager_);
                DEBUG_PRINTLN("Applied MapFusion to maps " + first_name + " and " + second_name);
                applied = true;
            }
        } else if (i + 2 < node.size()) {
            auto* mid_block = dynamic_cast<structured_control_flow::Block*>(&node.at(i + 1).first);
            if (mid_block && !is_malloc_block(*mid_block).empty()) {
                if (auto* second = dynamic_cast<structured_control_flow::StructuredLoop*>(&node.at(i + 2).first)) {
                    if (second->root().size() == 0) {
                        i++;
                        continue;
                    }
                    transformations::MapFusion transformation(*first, *second, false);
                    if (transformation.can_be_applied(builder_, analysis_manager_)) {
                        auto first_name = first->indvar()->get_name();
                        auto second_name = second->indvar()->get_name();
                        transformation.apply(builder_, analysis_manager_);
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
