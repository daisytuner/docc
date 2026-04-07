#pragma once

#include <string>
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class EinsumLift : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    EinsumLift(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "EinsumLift"; }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<EinsumLift> EinsumLiftPass;

class EinsumExtend : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    EinsumExtend(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "EinsumExtend"; }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<EinsumExtend> EinsumExtendPass;

class EinsumExpand : public visitor::StructuredSDFGVisitor {
public:
    EinsumExpand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "EinsumExpand"; }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<EinsumExpand> EinsumExpandPass;

class EinsumConversion : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    sdfg::PassReportConsumer* report_;

public:
    EinsumConversion(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        sdfg::PassReportConsumer* report = nullptr
    );

    static std::string name() { return "EinsumConversion"; }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<EinsumConversion> EinsumConversionPass;

class EinsumLower : public visitor::StructuredSDFGVisitor {
public:
    EinsumLower(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "EinsumLower"; }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<EinsumLower> EinsumLowerPass;

} // namespace passes
} // namespace sdfg
