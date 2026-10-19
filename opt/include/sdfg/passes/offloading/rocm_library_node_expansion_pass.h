#pragma once

#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class RocmExpansion : public visitor::NonStoppingStructuredSDFGVisitor {
    PassReportConsumer* report_ = nullptr;
    bool force_expand_;

public:
    RocmExpansion(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, const Options& options
    );

    void set_report(PassReportConsumer* report) {
        report_ = report;
    }

    static std::string name() {
        return "RocmExpansion";
    };

    bool accept(structured_control_flow::Block& node) override;
};

typedef VisitorPass<RocmExpansion> RocmExpansionPass;

} // namespace passes
} // namespace sdfg
