#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

/// Marks every CMathNode as approximate, permitting fast device intrinsics (e.g. __expf).
class ApproximateMath : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    ApproximateMath(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() {
        return "ApproximateMath";
    }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<ApproximateMath> ApproximateMathPass;

} // namespace passes
} // namespace sdfg
