#pragma once

#include <string>

#include "sdfg/users/user.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::users {

/// Traverses a structured SDFG and reports uses with the classic Users semantics:
/// one user per access node and direction (READ/WRITE/VIEW/MOVE), deduplicated symbol reads,
/// loop headers split into init / condition / update, and nothing after a Return.
/// Users of a node are reported contiguously between on_enter and on_leave of that node.
class LegacyUserVisitor : public visitor::ActualStructuredSDFGVisitor {
public:
    enum class LoopPart { None, Init, Condition, Update };

    using visitor::ActualStructuredSDFGVisitor::visit;

    /// Visits node and all of its children.
    void traverse(structured_control_flow::ControlFlowNode& node);

    bool visit(structured_control_flow::Block& node) override;
    bool visit(structured_control_flow::AssignmentBlock& node) override;
    bool visit(structured_control_flow::Sequence& node) override;
    bool visit(structured_control_flow::Return& node) override;
    bool visit(structured_control_flow::IfElse& node) override;
    bool handleStructuredLoop(structured_control_flow::StructuredLoop& loop) override;
    bool visit(structured_control_flow::While& node) override;
    bool visit(structured_control_flow::Continue& node) override;
    bool visit(structured_control_flow::Break& node) override;

protected:
    virtual void on_enter(structured_control_flow::ControlFlowNode& node) {
    }
    virtual void on_leave(structured_control_flow::ControlFlowNode& node) {
    }
    virtual void on_use(const std::string& container, Element* element, Use use, LoopPart part) = 0;
};

} // namespace sdfg::users
