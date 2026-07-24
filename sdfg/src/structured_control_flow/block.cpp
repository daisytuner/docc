#include "sdfg/structured_control_flow/block.h"

#include "sdfg/codegen/utils.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace structured_control_flow {

Block::Block(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {
    this->dataflow_ = std::make_unique<data_flow::DataFlowGraph>(this);
};

bool Block::accept(visitor::ActualStructuredSDFGVisitor& visitor) { return visitor.visit(*this); }

void Block::validate(const Function& function) const {
    this->dataflow_->validate(function);
    if (this->dataflow().get_parent() != this) {
        throw InvalidSDFGException("Block::validate: Dataflow parent does not point to self");
    }
};

const data_flow::DataFlowGraph& Block::dataflow() const { return *this->dataflow_; };

data_flow::DataFlowGraph& Block::dataflow() { return *this->dataflow_; };

void Block::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->dataflow_->replace(old_expression, new_expression);
}

void Block::replace(const symbolic::ExpressionMapping& replacements) { this->dataflow_->replace(replacements); }

AssignmentBlock::AssignmentBlock(
    ElementId element_id,
    const DebugInfo& debug_info,
    ControlFlowNode* parent,
    const control_flow::Assignments& assignments
)
    : ControlFlowNode(element_id, debug_info, parent), assignments_(assignments) {}

AssignmentBlock::AssignmentBlock(ElementId element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : ControlFlowNode(element_id, debug_info, parent) {}

bool AssignmentBlock::add_if_not_overwritten(const symbolic::Symbol& target, const symbolic::Expression& expr) {
    auto [it, was_added] = assignments_.insert({target, expr});
    return was_added;
}

bool AssignmentBlock::accept(visitor::ActualStructuredSDFGVisitor& visitor) { return visitor.visit(*this); }

void AssignmentBlock::validate(const Function& function) const {
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
}

const control_flow::Assignments& AssignmentBlock::assignments() const { return assignments_; }

control_flow::Assignments& AssignmentBlock::assignments() { return assignments_; }

bool AssignmentBlock::empty() const { return assignments_.empty(); }

size_t AssignmentBlock::size() const { return assignments_.size(); }

void AssignmentBlock::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
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
}

void AssignmentBlock::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& [old_expr, new_expr] : replacements) {
        if (SymEngine::is_a<SymEngine::Symbol>(*old_expr) && SymEngine::is_a<SymEngine::Symbol>(*new_expr)) {
            auto old_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expr);
            auto new_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expr);

            if (this->assignments().find(old_symbol) != this->assignments().end()) {
                this->assignments()[new_symbol] = this->assignments()[old_symbol];
                this->assignments().erase(old_symbol);
            }
        }
    }

    for (auto& entry : this->assignments()) {
        entry.second = symbolic::subs(entry.second, replacements);
    }
}

} // namespace structured_control_flow
} // namespace sdfg
