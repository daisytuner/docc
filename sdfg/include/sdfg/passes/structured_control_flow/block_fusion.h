#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class BlockFusion : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    /**
     * @brief Skips all blocks with libnodes in them, essentially treating all libnodes as side-effect nodes
     * Default false
     */
    bool ignore_libnodes_;

    bool can_be_applied(
        data_flow::DataFlowGraph& first_graph,
        control_flow::Assignments& first_assignments,
        data_flow::DataFlowGraph& second_graph,
        control_flow::Assignments& second_assignments
    );

    void apply(
        structured_control_flow::Block& first_block,
        control_flow::Assignments& first_assignments,
        structured_control_flow::Block& second_block,
        control_flow::Assignments& second_assignments
    );

public:
    BlockFusion(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        bool ignore_libnodes = false
    );

    static std::string name() { return "BlockFusion"; };

    bool accept(structured_control_flow::Sequence& node) override;
};

typedef VisitorPass<BlockFusion> BlockFusionPass;

class NoLibnodesBlockFusionPass : public Pass {
public:
    std::string name() override { return BlockFusion::name(); };

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override {
        BlockFusion visitor(builder, analysis_manager, true);
        return visitor.visit();
    };
};

} // namespace passes
} // namespace sdfg
