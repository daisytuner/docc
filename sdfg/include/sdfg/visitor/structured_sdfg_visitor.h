#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/reduce.h"

namespace sdfg {
namespace visitor {

/**
 * Very specific implementation, that technically visits the SDFG of an StructuredSDFGBuilder and not just any SDFG.
 * Also does some redundant / interleaved calling, where the visitor internally visits a sequence as a node and then
 * (non-recursively) each child of the sequence and of the child is itself a container, the body/sequence of the
 * container. This is quite far away from traditional visitor pattern.
 */
class StructuredSDFGVisitor {
protected:
    builder::StructuredSDFGBuilder& builder_;
    analysis::AnalysisManager& analysis_manager_;

    virtual bool visit_internal(structured_control_flow::Sequence& parent);

public:
    StructuredSDFGVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    virtual ~StructuredSDFGVisitor() = default;

    virtual bool visit();

    virtual bool accept(structured_control_flow::Block& node);

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

class NonStoppingStructuredSDFGVisitor : public StructuredSDFGVisitor {
private:
    bool applied_;

protected:
    bool visit_internal(structured_control_flow::Sequence& parent) override;

public:
    NonStoppingStructuredSDFGVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    bool visit() override;
};

/**
 * Visitor for Structured SDFG (not a builder).
 * follows the actual type hierarchy for the visit methods, but still does the hack of dyn-casting instead of an actual
 * visitor pattern, but this is hidden from users, just less efficient.
 *
 */
class ActualStructuredSDFGVisitor {
public:
    ActualStructuredSDFGVisitor();

    virtual ~ActualStructuredSDFGVisitor() = default;

    bool visit(sdfg::structured_control_flow::ControlFlowNode& node);

    bool dispatch(sdfg::structured_control_flow::ControlFlowNode& node); // return value really has no meaning yet.
                                                                         // Don't know how to do generic results
                                                                         // efficiently for C++ without making all the
                                                                         // actual node-types

    virtual bool visit(sdfg::structured_control_flow::Block& node);

    virtual bool visit(sdfg::structured_control_flow::Sequence& node);

    virtual bool visit(sdfg::structured_control_flow::Return& node);

    virtual bool visit(sdfg::structured_control_flow::IfElse& node);

    virtual bool visit(sdfg::structured_control_flow::For& node);

    virtual bool visit(sdfg::structured_control_flow::Map& node);

    virtual bool visit(sdfg::structured_control_flow::Reduce& node);

    virtual bool handleStructuredLoop(sdfg::structured_control_flow::StructuredLoop& loop);

    virtual bool visit(sdfg::structured_control_flow::While& node);

    virtual bool visit(sdfg::structured_control_flow::Continue& node);

    virtual bool visit(sdfg::structured_control_flow::Break& node);
};

} // namespace visitor
} // namespace sdfg
