#include "sdfg/einsum/passes/einsum_passes.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/einsum/einsum_detection.h"
#include "sdfg/einsum/einsum_node.h"
#include "sdfg/einsum/transformations/einsum2dot.h"
#include "sdfg/einsum/transformations/einsum2gemm.h"
#include "sdfg/einsum/transformations/einsum_extend.h"
#include "sdfg/einsum/transformations/einsum_lift.h"
#include "sdfg/einsum/transformations/einsum_promotion.h"
#include "sdfg/exceptions.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/types/scalar.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace einsum {

bool EinsumClusterDetectionPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    EinsumDetection detector;
    detector.run(builder.subject().root());
    auto& users = analysis_manager.get<analysis::Users>();
    std::unordered_set<const EinsumCluster*> shared_intermediates;
    for (auto& found : detector.einsums()) {
        for (auto* consumed : found->consumed_nodes) {
            auto* access = dynamic_cast<data_flow::AccessNode*>(consumed);
            if (!access) {
                continue;
            }
            // Local dataflow degree does not account for reads in other blocks.
            if (builder.subject().is_argument(access->data())) {
                shared_intermediates.insert(found.get());
            }
            for (auto* user : users.uses(access->data())) {
                if (user->element() != access) {
                    shared_intermediates.insert(found.get());
                }
            }
        }
    }
    bool applied = false;
    for (auto& found : detector.einsums()) {
        auto& cluster = *found;
        if (shared_intermediates.contains(&cluster)) {
            continue;
        }
        if (cluster.consumed_loops.empty() && cluster.consumed_nodes.size() == 1 &&
            dynamic_cast<EinsumNode*>(cluster.consumed_nodes.front())) {
            continue;
        }
        auto& original = *cluster.block;
        auto* block = &original;
        if (!cluster.consumed_loops.empty()) {
            auto& loop = *cluster.consumed_loops.front();
            auto& parent = static_cast<structured_control_flow::Sequence&>(*loop.get_parent());
            block = &builder.add_block_after(parent, loop, original.debug_info());
        }
        std::vector<data_flow::Subset> indices;
        for (size_t i = 0; i < cluster.inputs.size(); ++i) {
            indices.push_back(cluster.get_input_subset(i));
        }
        auto& node = builder.add_library_node<
            EinsumNode>(*block, original.debug_info(), cluster.inputs, cluster.dims, cluster.out_indices(), indices);
        for (size_t i = 0; i < cluster.inputs.size(); ++i) {
            auto* source = cluster.input_nodes.at(cluster.inputs[i]);
            if (!source) {
                // Subtraction is represented by a synthetic multiplicative -1.
                const types::Scalar type(cluster.output_edge->base_type().primitive_type());
                auto& constant = builder.add_constant(*block, "-1.0", type, original.debug_info());
                builder.add_computational_memlet(*block, constant, node, cluster.inputs[i], {}, type);
                continue;
            }
            auto& access = block == &original ? *source : builder.copy_node(*block, *source);
            builder.add_memlet(
                *block,
                access,
                "void",
                node,
                cluster.inputs[i],
                {},
                cluster.in_edges[i]->base_type(),
                original.debug_info()
            );
        }
        data_flow::AccessNode* read = nullptr;
        if (block == &original) {
            for (auto& edge : original.dataflow().in_edges(*cluster.consumed_nodes.front())) {
                if (auto* access = dynamic_cast<data_flow::AccessNode*>(&edge.src());
                    access && access->data() == cluster.output_node->data() && edge.subset() == cluster.out_indices()) {
                    read = access;
                    break;
                }
            }
        }
        if (!read) {
            read = &builder.add_access(*block, cluster.output_node->data(), original.debug_info());
        }
        auto& write = block == &original
                          ? *cluster.output_node
                          : builder.add_access(*block, cluster.output_node->data(), original.debug_info());
        auto& type = cluster.output_edge->base_type();
        builder.add_memlet(*block, *read, "void", node, "__einsum_out", {}, type, original.debug_info());
        builder.add_memlet(*block, node, "__einsum_out", write, "void", {}, type, original.debug_info());

        if (!cluster.consumed_loops.empty()) {
            auto& loop = *cluster.consumed_loops.front();
            auto& parent = static_cast<structured_control_flow::Sequence&>(*loop.get_parent());
            builder.remove_child(parent, parent.index(loop));
        } else {
            auto& dataflow = original.dataflow();
            for (auto* consumed : cluster.consumed_nodes) {
                while (dataflow.in_degree(*consumed)) {
                    builder.remove_memlet(original, *dataflow.in_edges(*consumed).begin());
                }
                while (dataflow.out_degree(*consumed)) {
                    builder.remove_memlet(original, *dataflow.out_edges(*consumed).begin());
                }
                builder.remove_node(original, *consumed);
            }
        }
        applied = true;
    }
    if (applied) {
        analysis_manager.invalidate_all();
    }
    return applied;
}

class BlockFinder : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    std::list<structured_control_flow::Block*> blocks_;

public:
    BlockFinder(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager), blocks_() {
    }

    static std::string name() {
        return "BlockFinder";
    }

    virtual bool accept(structured_control_flow::Block& block) {
        blocks_.push_back(&block);
        return true;
    }

    std::list<structured_control_flow::Block*>& blocks() {
        return blocks_;
    }
};

bool EinsumDetectionPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    BlockFinder block_finder(builder, analysis_manager);
    if (!block_finder.visit()) {
        // Fast exit because no blocks where found
        return false;
    }

    // Try lifting all available tasklets to einsum nodes and capture them
    bool applied = false;
    std::list<structured_control_flow::Block*> block_queue(block_finder.blocks());
    std::unordered_set<einsum::EinsumNode*> einsum_nodes;
    while (!block_queue.empty()) {
        structured_control_flow::Block* block = block_queue.front();
        block_queue.pop_front();

        // Find already existing einsum nodes
        auto libnodes = block->dataflow().library_nodes();
        for (auto* libnode : libnodes) {
            if (auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode)) {
                einsum_nodes.insert(einsum_node);
            }
        }

        // Lift tasklets to einsum node as far as possible
        auto tasklets = block->dataflow().tasklets();
        for (auto* tasklet : tasklets) {
            EinsumLift transformation(*tasklet);
            if (transformation.can_be_applied(builder, analysis_manager)) {
                transformation.apply(builder, analysis_manager);
                DEBUG_PRINTLN("Applied " << transformation.name());
                applied = true;

                // Re-visit the current block
                block_queue.push_front(block);
                break;
            }
        }
    }

    // Try extending all captured einsum nodes as much as possible
    std::list<einsum::EinsumNode*> einsum_queue(einsum_nodes.begin(), einsum_nodes.end());
    while (!einsum_queue.empty()) {
        einsum::EinsumNode* einsum_node = einsum_queue.front();
        einsum_queue.pop_front();

        // Extend einsum node as far as possible
        EinsumExtend transformation(*einsum_node);
        if (transformation.can_be_applied(builder, analysis_manager)) {
            einsum_nodes.erase(einsum_node);
            transformation.apply(builder, analysis_manager);
            DEBUG_PRINTLN("Applied " << transformation.name());
            applied = true;

            // Re-add and re-visit new einsum node
            auto* new_einsum_node = transformation.new_einsum_node();
            einsum_nodes.insert(new_einsum_node);
            einsum_queue.push_front(new_einsum_node);
        }
    }

    einsum_queue.insert(einsum_queue.end(), einsum_nodes.begin(), einsum_nodes.end());
    while (!einsum_queue.empty()) {
        einsum::EinsumNode* einsum_node = einsum_queue.front();
        einsum_queue.pop_front();

        EinsumPromotion transformation(*einsum_node);
        if (transformation.can_be_applied(builder, analysis_manager)) {
            transformation.apply(builder, analysis_manager);
            DEBUG_PRINTLN("Applied " << transformation.name());
            applied = true;

            // Re-visit new einsum node
            einsum_queue.push_front(transformation.new_einsum_node());
        }
    }

    return applied;
}

EinsumConversion::EinsumConversion(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    sdfg::PassReportConsumer* report
)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager), report_(report) {
}

bool EinsumConversion::accept(structured_control_flow::Block& block) {
    bool applied = false;

    for (auto* libnode : block.dataflow().library_nodes()) {
        if (auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode)) {
            Einsum2Dot dot_transformation(*einsum_node);
            if (report_) {
                dot_transformation.set_report(report_);
            }

            if (dot_transformation.can_be_applied(this->builder_, this->analysis_manager_)) {
                dot_transformation.apply(this->builder_, this->analysis_manager_);
                DEBUG_PRINTLN("Applied Einsum2Dot");
                applied = true;
                continue;
            }

            Einsum2Gemm gemm_transformation(*einsum_node);
            if (report_) {
                gemm_transformation.set_report(report_);
            }

            if (gemm_transformation.can_be_applied(this->builder_, this->analysis_manager_)) {
                gemm_transformation.apply(this->builder_, this->analysis_manager_);
                DEBUG_PRINTLN("Applied Einsum2Gemm");
                applied = true;
                continue;
            }
        }
    }

    return applied;
}

EinsumLower::EinsumLower(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
}

bool EinsumLower::accept(structured_control_flow::Block& block) {
    for (auto* libnode : block.dataflow().library_nodes()) {
        if (auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode)) {
            if (einsum_node->expand(this->builder_, this->analysis_manager_)) {
                DEBUG_PRINTLN("Applied EinsumLower");
                return true;
            } else {
                throw InvalidSDFGException("EinsumLower: Could not lower einsum node");
            }
        }
    }
    return false;
}

EinsumExpansion::EinsumExpansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
}

bool EinsumExpansion::accept(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();

    bool applied = false;
    for (auto* library_node : dataflow.library_nodes()) {
        if (library_node->implementation_type() != data_flow::ImplementationType_NONE) {
            continue;
        }

        if (library_node->code() == LibraryNodeType_Einsum) {
            auto enode = dynamic_cast<einsum::EinsumNode*>(library_node);
            if (enode->expand(this->builder_, this->analysis_manager_)) {
                return true;
            }
        }
    }
    return applied;
}

} // namespace einsum
} // namespace sdfg
