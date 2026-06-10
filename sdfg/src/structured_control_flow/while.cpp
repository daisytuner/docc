#include "sdfg/structured_control_flow/while.h"

namespace sdfg {
namespace structured_control_flow {

While::While(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {
    this->root_ = std::unique_ptr<Sequence>(new Sequence(++element_id, debug_info, this));
};

void While::validate(const Function& function) const { this->root_->validate(function); };

const Sequence& While::root() const { return *this->root_; };

Sequence& While::root() { return *this->root_; };

void While::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->root_->replace(old_expression, new_expression);
};

Break::Break(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {

      };

void Break::validate(const Function& function) const {};

void Break::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {

};

Continue::Continue(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {};

void Continue::validate(const Function& function) const {};

void Continue::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {

};

} // namespace structured_control_flow
} // namespace sdfg
