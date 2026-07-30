#pragma once

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/structured_data_flow_analysis.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {
namespace normalization {


class MapFusion : public visitor::NonStoppingStructuredSDFGVisitor {
    std::unique_ptr<analysis::LoopAnalysis> loop_analysis_;
    // When false, Case 2 (init-into-reduction hoisting) is disabled for this run.
    bool allow_init_hoist_;
    // Allow ProducerIntoConsumer fusing
    bool allow_prod_into_cons_;

public:
    MapFusion(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        bool allow_init_hoist = true,
        bool allow_prod_into_cons = true
    );

    static std::string name() { return "MapFusion"; };

    bool accept(structured_control_flow::Sequence& node) override;
};

// A VisitorPass-style wrapper that forwards the init-hoist flag to the visitor.
class MapFusionPass : public Pass {
    bool allow_init_hoist_;
    bool allow_prod_into_cons_;

public:
    MapFusionPass(bool allow_init_hoist = true, bool allow_prod_into_cons = true)
        : allow_init_hoist_(allow_init_hoist), allow_prod_into_cons_(allow_prod_into_cons) {}

    std::string name() override { return MapFusion::name(); }

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override {
        MapFusion visitor(builder, analysis_manager, allow_init_hoist_, allow_prod_into_cons_);
        return visitor.visit();
    }
};

} // namespace normalization
} // namespace passes
} // namespace sdfg
