#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"


namespace sdfg {
namespace passes {

class DotExpansion : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    DotExpansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() {
        return "DotExpansion";
    }
    virtual bool visit() override;

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<DotExpansion> DotExpansionPass;

} // namespace passes
} // namespace sdfg
