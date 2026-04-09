#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace rocm {

class RocmLibraryNodeTransferExtractionVisitor : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    RocmLibraryNodeTransferExtractionVisitor(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
    );

    static std::string name() { return "RocmLibraryNodeTransferExtractionPass"; }
    bool visit() override;
    bool accept(structured_control_flow::Block& block) override;
};

typedef passes::VisitorPass<RocmLibraryNodeTransferExtractionVisitor> RocmLibraryNodeTransferExtractionPass;

} // namespace rocm
} // namespace sdfg
