#include "sdfg/passes/gemm_expansion_pass.h"
#include <ostream>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"


namespace sdfg {
namespace passes {

GemmExpansion::GemmExpansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool GemmExpansion::visit() {
    DEBUG_PRINTLN("Running GemmExpansionPass on " << this->builder_.subject().name());
    return visitor::NonStoppingStructuredSDFGVisitor::visit();
}

bool GemmExpansion::accept(structured_control_flow::Block& block) {
    bool expanded = false;

    auto& scope_analysis = analysis_manager_.get<analysis::ScopeAnalysis>();
    for (auto lib_node : block.dataflow().library_nodes()) {
        if (lib_node->code() == math::blas::LibraryNodeType_GEMM) {
            auto gemm_node = static_cast<math::blas::GEMMNode*>(lib_node);
            if (symbolic::eq(gemm_node->m(), symbolic::one()) || symbolic::eq(gemm_node->n(), symbolic::one()) ||
                symbolic::eq(gemm_node->k(), symbolic::one())) {
                DEBUG_PRINTLN("found applicable GEMM");
                if (gemm_node->expand(builder_, analysis_manager_)) {
                    DEBUG_PRINTLN("expanded");
                    return true;
                }
            }
        }
    }

    return expanded;
}

} // namespace passes
} // namespace sdfg
