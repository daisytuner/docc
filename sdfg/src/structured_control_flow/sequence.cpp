#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/codegen/utils.h"

#include "sdfg/function.h"

namespace sdfg {
namespace structured_control_flow {

Transition::Transition(size_t element_id, const DebugInfo& debug_info, Sequence& parent)
    : Element(element_id, debug_info), parent_(&parent) {

      };

Transition::Transition(
    size_t element_id, const DebugInfo& debug_info, Sequence& parent, const control_flow::Assignments& assignments
)
    : Element(element_id, debug_info), parent_(&parent), assignments_(assignments) {

      };

void Transition::validate(const Function& function) const {
    for (const auto& entry : this->assignments_) {
        if (entry.first.is_null() || entry.second.is_null()) {
            throw InvalidSDFGException("Transition: Assignments cannot have null expressions");
        }
    }

    for (auto& entry : this->assignments_) {
        auto& lhs = entry.first;
        auto& type = function.type(lhs->get_name());
        if (type.type_id() == types::TypeID::Scalar) {
            if (!types::is_integer(type.primitive_type())) {
                throw InvalidSDFGException("Assignment - LHS: must be integer type");
            }
        } else if (type.type_id() == types::TypeID::Reference) {
            auto* reference = dynamic_cast<const sdfg::codegen::Reference*>(&type);
            assert(reference != nullptr);
            auto& referenced_type = reference->reference_type();
            if (referenced_type.type_id() != types::TypeID::Scalar) {
                throw InvalidSDFGException("Assignment - LHS: must be a reference to a scalar type");
            }
            if (!types::is_integer(referenced_type.primitive_type())) {
                throw InvalidSDFGException("Assignment - LHS: must be integer type");
            }
        } else {
            throw InvalidSDFGException("Assignment - LHS: must be scalar type or a reference thereof");
        }

        auto& rhs = entry.second;
        for (auto& atom : symbolic::atoms(rhs)) {
            if (symbolic::is_nullptr(atom)) {
                continue;
            }
            auto& atom_type = function.type(atom->get_name());

            // Scalar integers
            if (atom_type.type_id() == types::TypeID::Scalar) {
                if (!types::is_integer(atom_type.primitive_type())) {
                    throw InvalidSDFGException("Assignment - RHS: must evaluate to integer type");
                }
                continue;
            } else if (atom_type.type_id() == types::TypeID::Reference) {
                auto* reference = dynamic_cast<const sdfg::codegen::Reference*>(&atom_type);
                assert(reference != nullptr);
                auto& referenced_type = reference->reference_type();
                if (referenced_type.type_id() != types::TypeID::Scalar) {
                    throw InvalidSDFGException("Assignment - RHS: must be a reference to a scalar type");
                }
                if (!types::is_integer(referenced_type.primitive_type())) {
                    throw InvalidSDFGException("Assignment - RHS: must evaluate to integer type");
                }
                continue;
            } else if (atom_type.type_id() == types::TypeID::Pointer) {
                continue;
            } else {
                throw InvalidSDFGException("Assignment - RHS: must evaluate to integer or pointer type");
            }
        }
    }
};

const control_flow::Assignments& Transition::assignments() const { return this->assignments_; };

control_flow::Assignments& Transition::assignments() { return this->assignments_; };

Sequence& Transition::parent() { return *this->parent_; };

const Sequence& Transition::parent() const { return *this->parent_; };

bool Transition::empty() const { return this->assignments_.empty(); };

size_t Transition::size() const { return this->assignments_.size(); };

void Transition::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (SymEngine::is_a<SymEngine::Symbol>(*old_expression) && SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
        auto old_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expression);
        auto new_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);

        if (this->assignments().find(old_symbol) != this->assignments().end()) {
            this->assignments()[new_symbol] = this->assignments()[old_symbol];
            this->assignments().erase(old_symbol);
        }
    }

    for (auto& entry : this->assignments()) {
        entry.second = symbolic::subs(entry.second, old_expression, new_expression);
    }
};

Sequence::Sequence(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {

      };

void Sequence::validate(const Function& function) const {
    // children and transition have same length
    if (this->children_.size() != this->transitions_.size()) {
        throw InvalidSDFGException("Sequence must have the same number of children and transitions");
    }

    for (auto& child : this->children_) {
        child->validate(function);
    }
    for (auto& trans : this->transitions_) {
        trans->validate(function);
    }
};

size_t Sequence::size() const { return this->children_.size(); };

std::pair<const ControlFlowNode&, const Transition&> Sequence::at(size_t i) const {
    return {*this->children_.at(i), *this->transitions_.at(i)};
};

std::pair<ControlFlowNode&, Transition&> Sequence::at(size_t i) {
    return {*this->children_.at(i), *this->transitions_.at(i)};
};

int Sequence::index(const ControlFlowNode& child) const {
    for (size_t i = 0; i < this->children_.size(); i++) {
        if (this->children_.at(i).get() == &child) {
            return static_cast<int>(i);
        }
    }

    return -1;
};

int Sequence::index(const Transition& transition) const {
    for (size_t i = 0; i < this->transitions_.size(); i++) {
        if (this->transitions_.at(i).get() == &transition) {
            return static_cast<int>(i);
        }
    }

    return -1;
};

void Sequence::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& child : this->children_) {
        child->replace(old_expression, new_expression);
    }

    for (auto& trans : this->transitions_) {
        trans->replace(old_expression, new_expression);
    }
};

} // namespace structured_control_flow
} // namespace sdfg
