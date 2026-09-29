#include "sdfg/control_flow/interstate_edge.h"

#include "sdfg/function.h"

namespace sdfg {
namespace control_flow {

InterstateEdge::InterstateEdge(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Edge& edge,
    const control_flow::State& src,
    const control_flow::State& dst,
    const symbolic::Condition condition,
    const sdfg::control_flow::Assignments& assignments
)
    : Element(element_id, debug_info), edge_(edge), src_(src), dst_(dst), condition_(condition),
      assignments_(assignments) {

      };

void InterstateEdge::validate(const Function& function) const {
    for (auto& entry : this->assignments_) {
        auto& lhs = entry.first;
        auto& type = function.type(lhs->get_name());
        if (type.type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException("Assignment - LHS: must be scalar type");
        }
        if (!types::is_integer(type.primitive_type())) {
            throw InvalidSDFGException("Assignment - LHS: must be integer type");
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
            } else if (atom_type.type_id() == types::TypeID::Pointer) {
                continue;
            } else {
                throw InvalidSDFGException("Assignment - RHS: must evaluate to integer or pointer type");
            }
        }
    }

    if (this->condition_.is_null()) {
        throw InvalidSDFGException("InterstateEdge: Condition cannot be null");
    }
    if (!SymEngine::is_a_Boolean(*this->condition_)) {
        throw InvalidSDFGException("InterstateEdge: Condition must be a boolean expression");
    }
    for (auto& atom : symbolic::atoms(this->condition_)) {
        if (symbolic::is_nullptr(atom)) {
            continue;
        }
        auto& atom_type = function.type(atom->get_name());
        if (atom_type.type_id() != types::TypeID::Scalar && atom_type.type_id() != types::TypeID::Pointer) {
            throw InvalidSDFGException("Condition: must be integer type or pointer type");
        }
    }
};

const graph::Edge InterstateEdge::edge() const {
    return this->edge_;
};

const control_flow::State& InterstateEdge::src() const {
    return this->src_;
};

const control_flow::State& InterstateEdge::dst() const {
    return this->dst_;
};

const symbolic::Condition InterstateEdge::condition() const {
    return this->condition_;
};

bool InterstateEdge::is_unconditional() const {
    return symbolic::is_true(this->condition_);
};

const sdfg::control_flow::Assignments& InterstateEdge::assignments() const {
    return this->assignments_;
};

void InterstateEdge::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    symbolic::subs(this->condition_, old_expression, new_expression);

    if (SymEngine::is_a<SymEngine::Symbol>(*old_expression) && SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
        auto old_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(old_expression);
        if (this->assignments_.find(old_symbol) != this->assignments_.end()) {
            auto new_symbol = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
            this->assignments_[new_symbol] = this->assignments_[old_symbol];
            this->assignments_.erase(old_symbol);
        }
    }

    for (auto& entry : this->assignments_) {
        entry.second = symbolic::subs(entry.second, old_expression, new_expression);
    }
};

} // namespace control_flow
} // namespace sdfg
