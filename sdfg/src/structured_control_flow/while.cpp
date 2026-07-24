#include "sdfg/structured_control_flow/while.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace structured_control_flow {

While::While(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {
    this->root_ = std::unique_ptr<Sequence>(new Sequence(++element_id, debug_info, this));
};

bool While::accept(visitor::ActualStructuredSDFGVisitor& visitor) { return visitor.visit(*this); }

void While::validate(const Function& function) const { this->root_->validate(function); };

const Sequence& While::root() const { return *this->root_; };

Sequence& While::root() { return *this->root_; };

void While::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->root_->replace(old_expression, new_expression);
}

void While::replace(const symbolic::ExpressionMapping& replacements) { this->root_->replace(replacements); }

Break::Break(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {

      };

bool Break::accept(visitor::ActualStructuredSDFGVisitor& visitor) { return visitor.visit(*this); }

void Break::validate(const Function& function) const {};

void Break::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

void Break::replace(const symbolic::ExpressionMapping& replacements) {}

Continue::Continue(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {};

bool Continue::accept(visitor::ActualStructuredSDFGVisitor& visitor) { return visitor.visit(*this); }

void Continue::validate(const Function& function) const {};

void Continue::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

void Continue::replace(const symbolic::ExpressionMapping& replacements) {}

} // namespace structured_control_flow
} // namespace sdfg
