#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace rocm {

class RocblasBLASOffloadingExpansionVisitor : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    RocblasBLASOffloadingExpansionVisitor(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
    );

    static std::string name() { return "RocblasBLASOffloadingExpansionPass"; }
    bool visit() override;
    bool accept(structured_control_flow::Block& block) override;
};

typedef passes::VisitorPass<RocblasBLASOffloadingExpansionVisitor> RocblasBLASOffloadingExpansionPass;

} // namespace rocm
} // namespace sdfg
