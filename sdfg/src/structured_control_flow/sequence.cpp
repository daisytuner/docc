#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

#include "sdfg/function.h"

namespace sdfg {
namespace structured_control_flow {

Sequence::Sequence(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {

      };

bool Sequence::accept(visitor::ActualStructuredSDFGVisitor& visitor) { return visitor.visit(*this); }

void Sequence::validate(const Function& function) const {
    for (auto& child : this->children_) {
        child->validate(function);
    }
};

size_t Sequence::size() const { return this->children_.size(); };

const ControlFlowNode& Sequence::at(size_t i) const { return *this->children_.at(i); }

ControlFlowNode& Sequence::at(size_t i) { return *this->children_.at(i); }

int Sequence::index(const ControlFlowNode& child) const {
    for (size_t i = 0; i < this->children_.size(); i++) {
        if (this->children_.at(i).get() == &child) {
            return static_cast<int>(i);
        }
    }

    return -1;
};

void Sequence::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& child : this->children_) {
        child->replace(old_expression, new_expression);
    }
}

void Sequence::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& child : this->children_) {
        child->replace(replacements);
    }
};

} // namespace structured_control_flow
} // namespace sdfg
