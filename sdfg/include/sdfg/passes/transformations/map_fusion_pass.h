#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class MapFusion : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    MapFusion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "MapFusion"; };

    bool accept(structured_control_flow::Sequence& node) override;
};

typedef VisitorPass<MapFusion> MapFusionPass;

} // namespace passes
} // namespace sdfg
