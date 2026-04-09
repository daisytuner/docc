#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace cuda {

class CudaLibraryNodeTransferExtractionVisitor : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    CudaLibraryNodeTransferExtractionVisitor(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
    );

    static std::string name() { return "CudaLibraryNodeTransferExtractionPass"; }
    bool visit() override;
    bool accept(structured_control_flow::Block& block) override;
};

typedef passes::VisitorPass<CudaLibraryNodeTransferExtractionVisitor> CudaLibraryNodeTransferExtractionPass;

} // namespace cuda
} // namespace sdfg
