#include "sdfg/passes/dot_expansion_pass.h"

#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/while.h"


namespace sdfg {
namespace passes {

DotExpansion::DotExpansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool DotExpansion::visit() {
    DEBUG_PRINTLN("Running DotExpansionPass on " << this->builder_.subject().name());
    return visitor::NonStoppingStructuredSDFGVisitor::visit();
}

bool DotExpansion::accept(structured_control_flow::Block& block) {
    bool expanded = false;

    for (auto lib_node : block.dataflow().library_nodes()) {
        if (lib_node->code() == math::blas::LibraryNodeType_DOT) {
            auto dot_node = static_cast<math::blas::DotNode*>(lib_node);
            if (dot_node->expand(builder_, analysis_manager_)) {
                return true;
            }
        }
    }

    return expanded;
}

} // namespace passes
} // namespace sdfg
