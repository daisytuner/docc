#include "sdfg/structured_control_flow/if_else.h"

#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace structured_control_flow {

IfElse::IfElse(size_t element_id, const DebugInfo& debug_info)
    : ControlFlowNode(element_id, debug_info) {

      };

void IfElse::validate(const Function& function) const {
    for (auto& entry : this->cases_) {
        entry->validate(function);
    }
    for (auto& entry : this->conditions_) {
        if (entry.is_null()) {
            throw InvalidSDFGException("IfElse: Conditions cannot be null");
        }
        if (!SymEngine::is_a_Boolean(*entry)) {
            throw InvalidSDFGException("IfElse: Conditions must be boolean expressions");
        }
    }
    if (this->cases_.size() != this->conditions_.size()) {
        throw InvalidSDFGException("IfElse: Number of cases must be equal to number of conditions");
    }
};

size_t IfElse::size() const { return this->cases_.size(); };

std::pair<const Sequence&, const symbolic::Condition> IfElse::at(size_t i) const {
    return {*this->cases_.at(i), this->conditions_.at(i)};
};

std::pair<Sequence&, const symbolic::Condition> IfElse::at(size_t i) {
    return {*this->cases_.at(i), this->conditions_.at(i)};
};

bool IfElse::is_complete() const {
    auto condition = symbolic::__false__();
    for (auto& entry : this->conditions_) {
        condition = symbolic::Or(condition, entry);
    }
    if (symbolic::is_true(condition)) return true;

    symbolic::CNF cnf_cond = symbolic::conjunctive_normal_form(condition);
    for (auto& clause : cnf_cond) {
        if (!symbolic::is_tautology(clause)) {
            return false;
        }
    }
    return true;
};

void IfElse::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (size_t i = 0; i < this->cases_.size(); ++i) {
        this->cases_.at(i)->replace(old_expression, new_expression);
        this->conditions_.at(i) = symbolic::subs(this->conditions_.at(i), old_expression, new_expression);
    }
};

} // namespace structured_control_flow
} // namespace sdfg
