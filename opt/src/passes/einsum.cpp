#include "sdfg/passes/einsum.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/exceptions.h"
#include "sdfg/helpers/helpers.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/transformations/einsum2dot.h"
#include "sdfg/transformations/einsum2gemm.h"
#include "sdfg/transformations/einsum_expand.h"
#include "sdfg/transformations/einsum_extend.h"
#include "sdfg/transformations/einsum_lift.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

EinsumLift::EinsumLift(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool EinsumLift::accept(structured_control_flow::Block& block) {
    bool applied = false;

    auto tasklets = block.dataflow().tasklets();
    for (auto* tasklet : tasklets) {
        transformations::EinsumLift transformation(*tasklet);
        if (transformation.can_be_applied(this->builder_, this->analysis_manager_)) {
            transformation.apply(this->builder_, this->analysis_manager_);
            DEBUG_PRINTLN("Applied EinsumLift");
            applied = true;
        }
    }

    return applied;
}

EinsumExtend::EinsumExtend(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool EinsumExtend::accept(structured_control_flow::Block& block) {
    for (auto* libnode : block.dataflow().library_nodes()) {
        if (auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode)) {
            transformations::EinsumExtend transformation(*einsum_node);
            if (transformation.can_be_applied(this->builder_, this->analysis_manager_)) {
                transformation.apply(this->builder_, this->analysis_manager_);
                DEBUG_PRINTLN("Applied EinsumExtend");
                return true;
            }
        }
    }
    return false;
}

EinsumExpand::EinsumExpand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::StructuredSDFGVisitor(builder, analysis_manager) {}

bool EinsumExpand::accept(structured_control_flow::Block& block) {
    for (auto* libnode : block.dataflow().library_nodes()) {
        if (auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode)) {
            transformations::EinsumExpand transformation(*einsum_node);
            if (transformation.can_be_applied(this->builder_, this->analysis_manager_)) {
                transformation.apply(this->builder_, this->analysis_manager_);
                DEBUG_PRINTLN("Applied EinsumExpand");
                return true;
            }
        }
    }
    return false;
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
        if (auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode)) {
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

} // namespace passes
} // namespace sdfg
