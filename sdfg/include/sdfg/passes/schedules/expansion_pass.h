/**
 * @file expansion_pass.h
 * @brief Library node expansion pass
 *
 * This file defines the expansion pass that transforms library nodes with
 * ImplementationType_NONE into primitive SDFG operations. The expansion pass
 * visits all blocks in the SDFG and attempts to expand math library nodes.
 *
 * ## Expansion Process
 *
 * The expansion pass:
 * 1. Iterates through all library nodes in each block
 * 2. Skips nodes with a specific implementation type (not NONE)
 * 3. For math nodes with ImplementationType_NONE, calls their expand() method
 * 4. The expand() method transforms the high-level operation into maps, tasklets, etc.
 *
 * The pass is applied iteratively until no more expansions occur, allowing
 * multi-level expansions where one library node expands into others.
 *
 * @see math::MathNode::expand for node-specific expansion logic
 * @see passes::Pass for the pass interface
 */

#pragma once

#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class MathExpansionPass;

class MathExpansionVisitor : public visitor::ActualStructuredSDFGVisitor {
    friend MathExpansionPass;

private:
    builder::StructuredSDFGBuilder& builder_;
    analysis::AnalysisManager& analysis_manager_;

    struct LibNodeContainer {
        math::MathNode& node;
        structured_control_flow::Block& block;
    };

    std::vector<LibNodeContainer> nodes_to_expand_;

public:
    /**
     * @brief Construct the expansion visitor
     * @param builder SDFG builder for creating new nodes
     * @param analysis_manager Analysis manager for querying properties
     */
    MathExpansionVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    bool visit(sdfg::structured_control_flow::Block& node) override;
};

/**
 * @class MathExpansionPass
 * @brief Looks for and expands math-nodes that are not already mapped to a specific target
 */
class MathExpansionPass : public Pass {
    std::string name() override { return "MathExpansion"; }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
