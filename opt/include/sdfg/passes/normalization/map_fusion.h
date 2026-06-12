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

public:
    MapFusion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "MapFusion"; };

    bool accept(structured_control_flow::Sequence& node) override;
};

typedef VisitorPass<MapFusion> MapFusionPass;

} // namespace normalization
} // namespace passes
} // namespace sdfg
