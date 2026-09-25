#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class TrivialReferenceConversion : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    TrivialReferenceConversion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() {
        return "TrivialReferenceConversion";
    }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<TrivialReferenceConversion> TrivialReferenceConversionPass;

} // namespace passes
} // namespace sdfg
