#include "sdfg/passes/schedules/expansion_pass.h"

#include "sdfg/data_flow/library_nodes/math/math.h"

namespace sdfg {
namespace passes {

MathExpansionVisitor::
    MathExpansionVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::ActualStructuredSDFGVisitor(), builder_(builder), analysis_manager_(analysis_manager) {}

bool MathExpansionVisitor::visit(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();


    for (auto* library_node : dataflow.library_nodes()) {
        if (library_node->implementation_type() != data_flow::ImplementationType_NONE) {
            continue;
        }

        if (auto math_node = dynamic_cast<math::MathNode*>(library_node)) {
            this->nodes_to_expand_.emplace_back(*math_node, node);
        }
    }
    return true;
}

bool MathExpansionPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    MathExpansionVisitor v(builder, analysis_manager);

    v.dispatch(builder.subject().root());

    auto& nodes = v.nodes_to_expand_;

    bool expanded_any = false;

    for (auto& entry : std::views::reverse(nodes)) {
        // TODO: check if the prerequisits are met, like if the libNode is standalone or if we need to cut it out of a
        // larger block first

        if (entry.node.expand(builder, analysis_manager)) {
            // If expansion was successful, remove the original library node // TODO requires new API to do this clean
            // builder.remove_node(entry.block, entry.node);
            // remove block
            expanded_any |= true;
        }
    }

    return expanded_any;
}
} // namespace passes
} // namespace sdfg
