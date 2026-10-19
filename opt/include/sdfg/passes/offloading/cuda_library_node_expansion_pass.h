#pragma once

#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class CudaExpansion : public visitor::NonStoppingStructuredSDFGVisitor {
    PassReportConsumer* report_ = nullptr;
    bool force_expand_;

public:
    CudaExpansion(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const Options& options = Options::empty()
    );

    void set_report(PassReportConsumer* report) {
        report_ = report;
    }

    static std::string name() {
        return "CudaExpansion";
    };

    bool accept(structured_control_flow::Block& node) override;
};

typedef VisitorPass<CudaExpansion> CudaExpansionPass;

} // namespace passes
} // namespace sdfg
