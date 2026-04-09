#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace cuda {

class CublasBLASOffloadingExpansionVisitor : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    CublasBLASOffloadingExpansionVisitor(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
    );

    static std::string name() { return "CublasBLASOffloadingExpansionPass"; }
    bool visit() override;
    bool accept(structured_control_flow::Block& block) override;
};

typedef passes::VisitorPass<CublasBLASOffloadingExpansionVisitor> CublasBLASOffloadingExpansionPass;

} // namespace cuda
} // namespace sdfg
