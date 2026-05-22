#include "sdfg/passes/einsum.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/einsum_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/transformations/einsum2dot.h"
#include "sdfg/transformations/einsum2gemm.h"
#include "sdfg/transformations/einsum_extend.h"
#include "sdfg/transformations/einsum_lift.h"
#include "sdfg/transformations/einsum_promotion.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class BlockFinder : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    std::list<structured_control_flow::Block*> blocks_;

public:
    BlockFinder(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager), blocks_() {}

    static std::string name() { return "BlockFinder"; }

    virtual bool accept(structured_control_flow::Block& block) {
        blocks_.push_back(&block);
        return true;
    }

    std::list<structured_control_flow::Block*>& blocks() { return blocks_; }
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
    std::unordered_set<math::tensor::EinsumNode*> einsum_nodes;
    while (!block_queue.empty()) {
        structured_control_flow::Block* block = block_queue.front();
        block_queue.pop_front();

        // Find already existing einsum nodes
        auto libnodes = block->dataflow().library_nodes();
        for (auto* libnode : libnodes) {
            if (auto* einsum_node = dynamic_cast<math::tensor::EinsumNode*>(libnode)) {
                einsum_nodes.insert(einsum_node);
            }
        }

        // Lift tasklets to einsum node as far as possible
        auto tasklets = block->dataflow().tasklets();
        for (auto* tasklet : tasklets) {
            transformations::EinsumLift transformation(*tasklet);
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
    std::list<math::tensor::EinsumNode*> einsum_queue(einsum_nodes.begin(), einsum_nodes.end());
    while (!einsum_queue.empty()) {
        math::tensor::EinsumNode* einsum_node = einsum_queue.front();
        einsum_queue.pop_front();

        // Extend einsum node as far as possible
        transformations::EinsumExtend transformation(*einsum_node);
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
        math::tensor::EinsumNode* einsum_node = einsum_queue.front();
        einsum_queue.pop_front();

        transformations::EinsumPromotion transformation(*einsum_node);
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
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager), report_(report) {}

bool EinsumConversion::accept(structured_control_flow::Block& block) {
    bool applied = false;

    for (auto* libnode : block.dataflow().library_nodes()) {
        if (auto* einsum_node = dynamic_cast<math::tensor::EinsumNode*>(libnode)) {
            transformations::Einsum2Dot dot_transformation(*einsum_node);
            if (report_) {
                dot_transformation.set_report(report_);
            }

            if (dot_transformation.can_be_applied(this->builder_, this->analysis_manager_)) {
                dot_transformation.apply(this->builder_, this->analysis_manager_);
                DEBUG_PRINTLN("Applied Einsum2Dot");
                applied = true;
                continue;
            }

            transformations::Einsum2Gemm gemm_transformation(*einsum_node);
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
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool EinsumLower::accept(structured_control_flow::Block& block) {
    for (auto* libnode : block.dataflow().library_nodes()) {
        if (auto* einsum_node = dynamic_cast<math::tensor::EinsumNode*>(libnode)) {
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

} // namespace passes
} // namespace sdfg
