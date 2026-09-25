#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class TensorToPointerConversion : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    TensorToPointerConversion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() {
        return "TensorToPointerConversion";
    }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<TensorToPointerConversion> TensorToPointerConversionPass;

} // namespace passes
} // namespace sdfg
