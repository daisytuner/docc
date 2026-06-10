#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class TypeMinimization : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    bool is_safe_trunc(symbolic::Expression expr, symbolic::BoundAnalysis& ba_tight, symbolic::BoundAnalysis& ba_loose);

public:
    TypeMinimization(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "TypeMinimization"; }

    virtual bool accept(structured_control_flow::Block& block) override;

    virtual bool accept(structured_control_flow::For& loop) override;
};

typedef VisitorPass<TypeMinimization> TypeMinimizationPass;

} // namespace passes
} // namespace sdfg
