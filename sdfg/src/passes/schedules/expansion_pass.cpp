#include "sdfg/passes/schedules/expansion_pass.h"

#include "sdfg/data_flow/library_nodes/math/math.h"

namespace sdfg {
namespace passes {

Expansion::Expansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool Expansion::accept(structured_control_flow::Block& node) {
    auto& dataflow = node.dataflow();

    bool applied = false;
    for (auto* library_node : dataflow.library_nodes()) {
        if (library_node->implementation_type() != data_flow::ImplementationType_NONE) {
            continue;
        }

        if (auto math_node = dynamic_cast<math::MathNode*>(library_node)) {
            if (math_node->expand(this->builder_, this->analysis_manager_)) {
                return true;
            }
        }
    }
    return applied;
}

} // namespace passes
} // namespace sdfg
