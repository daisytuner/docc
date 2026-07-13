#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace visitor {

class ImmutableStructuredSDFGVisitor {
protected:
    StructuredSDFG& sdfg_;
    analysis::AnalysisManager& analysis_manager_;

    virtual bool visit_internal(structured_control_flow::Sequence& parent);

public:
    ImmutableStructuredSDFGVisitor(StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager);

    virtual ~ImmutableStructuredSDFGVisitor() = default;

    virtual bool visit();

    virtual bool accept(structured_control_flow::Block& node);

    virtual bool accept(structured_control_flow::AssignmentBlock& node);

    virtual bool accept(structured_control_flow::Sequence& node);

    virtual bool accept(structured_control_flow::Return& node);

    virtual bool accept(structured_control_flow::IfElse& node);

    virtual bool accept(structured_control_flow::For& node);

    virtual bool accept(structured_control_flow::While& node);

    virtual bool accept(structured_control_flow::Continue& node);

    virtual bool accept(structured_control_flow::Break& node);

    virtual bool accept(structured_control_flow::Map& node);

    virtual bool accept(structured_control_flow::Reduce& node);
};

} // namespace visitor
} // namespace sdfg
