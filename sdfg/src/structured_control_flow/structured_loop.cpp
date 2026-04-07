#include "sdfg/structured_control_flow/structured_loop.h"

#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/polynomials.h"

namespace sdfg {
namespace structured_control_flow {

StructuredLoop::StructuredLoop(
    size_t element_id,
    const DebugInfo& debug_info,
    symbolic::Symbol indvar,
    symbolic::Expression init,
    symbolic::Expression update,
    symbolic::Condition condition
)
    : ControlFlowNode(element_id, debug_info), indvar_(indvar), init_(init), update_(update), condition_(condition) {
    this->root_ = std::unique_ptr<Sequence>(new Sequence(++element_id, debug_info));
}

void StructuredLoop::validate(const Function& function) const {
    if (this->indvar_.is_null()) {
        throw InvalidSDFGException("StructuredLoop: Induction variable cannot be null");
    }
    if (this->init_.is_null()) {
        throw InvalidSDFGException("StructuredLoop: Initialization expression cannot be null");
    }
    if (this->update_.is_null()) {
        throw InvalidSDFGException("StructuredLoop: Update expression cannot be null");
    }
    if (this->condition_.is_null()) {
        throw InvalidSDFGException("StructuredLoop: Condition expression cannot be null");
    }
    if (!SymEngine::is_a_Boolean(*this->condition_)) {
        throw InvalidSDFGException("StructuredLoop: Condition expression must be a boolean expression");
    }

    this->root_->validate(function);
};

const symbolic::Symbol StructuredLoop::indvar() const { return this->indvar_; };

const symbolic::Expression StructuredLoop::init() const { return this->init_; };

const symbolic::Expression StructuredLoop::update() const { return this->update_; };

const symbolic::Condition StructuredLoop::condition() const { return this->condition_; };

Sequence& StructuredLoop::root() const { return *this->root_; };

void StructuredLoop::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    if (symbolic::eq(this->indvar_, old_expression) && SymEngine::is_a<SymEngine::Symbol>(*new_expression)) {
        this->indvar_ = SymEngine::rcp_static_cast<const SymEngine::Symbol>(new_expression);
    }
    this->init_ = symbolic::subs(this->init_, old_expression, new_expression);
    this->update_ = symbolic::subs(this->update_, old_expression, new_expression);
    this->condition_ = symbolic::subs(this->condition_, old_expression, new_expression);

    this->root_->replace(old_expression, new_expression);
};

symbolic::Integer StructuredLoop::stride() {
    auto expr = this->update();
    auto indvar = this->indvar();

    symbolic::SymbolVec gens = {indvar};
    auto polynomial = symbolic::polynomial(expr, gens);
    if (polynomial.is_null()) {
        return SymEngine::null;
    }
    auto coeffs = symbolic::affine_coefficients(polynomial, gens);
    if (coeffs.empty()) {
        return SymEngine::null;
    }
    if (coeffs.size() > 2 || coeffs.find(indvar) == coeffs.end() ||
        coeffs.find(symbolic::symbol("__daisy_constant__")) == coeffs.end()) {
        return SymEngine::null;
    }

    // Exponential strides (e.g., i = i * 2) are not supported, so the coefficient must be a positive integer
    auto mul_coeff = coeffs.at(indvar);
    if (!SymEngine::is_a<SymEngine::Integer>(*mul_coeff)) {
        return SymEngine::null;
    }
    auto int_mul_coeff = SymEngine::rcp_static_cast<const SymEngine::Integer>(mul_coeff)->as_int();
    if (int_mul_coeff != 1) {
        return SymEngine::null;
    }

    auto add_coeff = coeffs.at(symbolic::symbol("__daisy_constant__"));
    if (!SymEngine::is_a<SymEngine::Integer>(*add_coeff)) {
        return SymEngine::null;
    }
    auto int_add_coeff = SymEngine::rcp_static_cast<const SymEngine::Integer>(add_coeff)->as_int();
    return SymEngine::integer(int_add_coeff);
};

bool StructuredLoop::is_contiguous() {
    auto stride = this->stride();
    if (stride.is_null()) {
        return false;
    }
    auto stride_int = stride->as_int();
    return stride_int == 1;
}

bool StructuredLoop::is_monotonic() {
    auto stride = this->stride();
    if (stride.is_null()) {
        return false;
    }
    auto stride_int = stride->as_int();
    return stride_int > 0;
}

symbolic::Expression StructuredLoop::canonical_bound() {
    auto stride = this->stride();
    if (stride.is_null()) {
        return SymEngine::null;
    }
    auto stride_int = stride->as_int();
    if (stride_int == 0 || stride_int > 1 || stride_int < -1) {
        return SymEngine::null;
    }
    if (stride_int < 0) {
        return this->canonical_bound_lower();
    } else {
        return this->canonical_bound_upper();
    }
}

symbolic::Expression StructuredLoop::canonical_bound_upper() {
    symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(condition_);
    } catch (...) {
        return SymEngine::null;
    }

    symbolic::Expression min_bound = SymEngine::null;
    for (const auto& clause : cnf) {
        // For upper bound extraction, we require unit clauses (single literal per clause)
        // Multi-clause disjunctions like (i < N || i < M) are not supported
        if (clause.size() != 1) {
            return SymEngine::null;
        }

        auto literal = clause[0];
        symbolic::Expression bound = SymEngine::null;
        if (!symbolic::uses(literal, indvar_)) {
            // Dead check
            continue;
        }

        if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
            // Handle: lhs < rhs
            auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
            auto lhs = lt->get_arg1();
            auto rhs = lt->get_arg2();

            // Check if indvar is on LHS
            if (!symbolic::uses(lhs, indvar_->get_name())) {
                // indvar not on LHS - this is a lower bound constraint, skip it
                continue;
            }
            if (symbolic::uses(rhs, indvar_->get_name())) {
                // indvar on both sides, can't extract
                return SymEngine::null;
            }

            // Extract: coeff * indvar + offset < rhs  =>  indvar < (rhs - offset) / coeff
            symbolic::SymbolVec syms = {indvar_};
            auto poly = symbolic::polynomial(lhs, syms);
            if (poly.is_null()) {
                return SymEngine::null;
            }
            auto coeffs = symbolic::affine_coefficients(poly, syms);
            if (coeffs.empty() || coeffs.find(indvar_) == coeffs.end()) {
                return SymEngine::null;
            }

            auto coeff = coeffs.at(indvar_);
            symbolic::Expression offset = symbolic::zero();
            if (coeffs.count(symbolic::symbol("__daisy_constant__"))) {
                offset = coeffs.at(symbolic::symbol("__daisy_constant__"));
            }

            // Coefficient must be a positive integer for upper bound
            if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
                return SymEngine::null;
            }
            auto coeff_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(coeff)->as_int();
            if (coeff_int <= 0) {
                return SymEngine::null;
            }

            // bound = (rhs - offset) / coeff
            bound = symbolic::expand(symbolic::sub(rhs, offset));
            if (coeff_int != 1) {
                bound = symbolic::expand(symbolic::div(bound, coeff));
            }

        } else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
            // Handle: lhs <= rhs  =>  lhs < rhs + 1
            auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(literal);
            auto lhs = le->get_arg1();
            auto rhs = le->get_arg2();

            if (!symbolic::uses(lhs, indvar_->get_name())) {
                // indvar not on LHS - this is a lower bound constraint, skip it
                continue;
            }
            if (symbolic::uses(rhs, indvar_->get_name())) {
                return SymEngine::null;
            }

            symbolic::SymbolVec syms = {indvar_};
            auto poly = symbolic::polynomial(lhs, syms);
            if (poly.is_null()) {
                return SymEngine::null;
            }
            auto coeffs = symbolic::affine_coefficients(poly, syms);
            if (coeffs.empty() || coeffs.find(indvar_) == coeffs.end()) {
                return SymEngine::null;
            }

            auto coeff = coeffs.at(indvar_);
            symbolic::Expression offset = symbolic::zero();
            if (coeffs.count(symbolic::symbol("__daisy_constant__"))) {
                offset = coeffs.at(symbolic::symbol("__daisy_constant__"));
            }

            if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
                return SymEngine::null;
            }
            auto coeff_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(coeff)->as_int();
            if (coeff_int <= 0) {
                return SymEngine::null;
            }

            // bound = (rhs + 1 - offset) / coeff
            bound = symbolic::expand(symbolic::sub(symbolic::add(rhs, symbolic::one()), offset));
            if (coeff_int != 1) {
                bound = symbolic::expand(symbolic::div(bound, coeff));
            }

        } else {
            // Other comparison types don't give upper bounds
            return SymEngine::null;
        }

        if (bound != SymEngine::null) {
            if (min_bound.is_null()) {
                min_bound = bound;
            } else {
                min_bound = symbolic::min(min_bound, bound);
            }
        }
    }
    return min_bound;
}

symbolic::Expression StructuredLoop::canonical_bound_lower() {
    symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(condition_);
    } catch (...) {
        return SymEngine::null;
    }

    symbolic::Expression max_bound = SymEngine::null;
    for (const auto& clause : cnf) {
        // For lower bound extraction, we require unit clauses
        if (clause.size() != 1) {
            return SymEngine::null;
        }

        auto literal = clause[0];
        symbolic::Expression bound = SymEngine::null;
        if (!symbolic::uses(literal, indvar_)) {
            // Dead check
            continue;
        }

        if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
            // Handle: lhs < rhs where rhs contains indvar => indvar > lhs (lower bound)
            auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(literal);
            auto lhs = lt->get_arg1();
            auto rhs = lt->get_arg2();

            // For lower bound, indvar should be on RHS: bound < indvar
            if (!symbolic::uses(rhs, indvar_->get_name())) {
                // indvar not on RHS - this is an upper bound constraint, skip it
                continue;
            }
            if (symbolic::uses(lhs, indvar_->get_name())) {
                return SymEngine::null;
            }

            // Extract: lhs < coeff * indvar + offset  =>  indvar > (lhs - offset) / coeff
            symbolic::SymbolVec syms = {indvar_};
            auto poly = symbolic::polynomial(rhs, syms);
            if (poly.is_null()) {
                return SymEngine::null;
            }
            auto coeffs = symbolic::affine_coefficients(poly, syms);
            if (coeffs.empty() || coeffs.find(indvar_) == coeffs.end()) {
                return SymEngine::null;
            }

            auto coeff = coeffs.at(indvar_);
            symbolic::Expression offset = symbolic::zero();
            if (coeffs.count(symbolic::symbol("__daisy_constant__"))) {
                offset = coeffs.at(symbolic::symbol("__daisy_constant__"));
            }

            if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
                return SymEngine::null;
            }
            auto coeff_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(coeff)->as_int();
            if (coeff_int <= 0) {
                return SymEngine::null;
            }

            // bound = (lhs - offset) / coeff
            bound = symbolic::expand(symbolic::sub(lhs, offset));
            if (coeff_int != 1) {
                bound = symbolic::expand(symbolic::div(bound, coeff));
            }

        } else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
            // Handle: lhs <= rhs where rhs contains indvar => indvar >= lhs => indvar > lhs - 1
            auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(literal);
            auto lhs = le->get_arg1();
            auto rhs = le->get_arg2();

            if (!symbolic::uses(rhs, indvar_->get_name())) {
                // indvar not on RHS - this is an upper bound constraint, skip it
                continue;
            }
            if (symbolic::uses(lhs, indvar_->get_name())) {
                return SymEngine::null;
            }

            symbolic::SymbolVec syms = {indvar_};
            auto poly = symbolic::polynomial(rhs, syms);
            if (poly.is_null()) {
                return SymEngine::null;
            }
            auto coeffs = symbolic::affine_coefficients(poly, syms);
            if (coeffs.empty() || coeffs.find(indvar_) == coeffs.end()) {
                return SymEngine::null;
            }

            auto coeff = coeffs.at(indvar_);
            symbolic::Expression offset = symbolic::zero();
            if (coeffs.count(symbolic::symbol("__daisy_constant__"))) {
                offset = coeffs.at(symbolic::symbol("__daisy_constant__"));
            }

            if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
                return SymEngine::null;
            }
            auto coeff_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(coeff)->as_int();
            if (coeff_int <= 0) {
                return SymEngine::null;
            }

            // bound = (lhs - 1 - offset) / coeff
            bound = symbolic::expand(symbolic::sub(symbolic::sub(lhs, symbolic::one()), offset));
            if (coeff_int != 1) {
                bound = symbolic::expand(symbolic::div(bound, coeff));
            }

        } else {
            return SymEngine::null;
        }

        if (bound != SymEngine::null) {
            if (max_bound.is_null()) {
                max_bound = bound;
            } else {
                max_bound = symbolic::max(max_bound, bound);
            }
        }
    }
    return max_bound;
}

symbolic::Expression StructuredLoop::num_iterations() {
    // implies |stride| == 1, so we can compute number of iterations as (bound - init)
    auto bound = this->canonical_bound();
    if (bound.is_null()) {
        return SymEngine::null;
    }
    auto num_iters = symbolic::expand(symbolic::sub(bound, this->init()));
    num_iters = symbolic::simplify(num_iters);
    return num_iters;
}

bool StructuredLoop::is_loop_normal_form() {
    // Check if init is zero
    if (!symbolic::eq(this->init_, symbolic::zero())) {
        return false;
    }

    // Check if it has positive unit stride
    auto stride = this->stride();
    if (stride.is_null() || stride->as_int() != 1) {
        return false;
    }

    // Check if condition has a canonical bound
    auto bound = this->canonical_bound();
    if (bound.is_null()) {
        return false;
    }

    return true;
}

} // namespace structured_control_flow
} // namespace sdfg
