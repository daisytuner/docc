#include "sdfg/symbolic/delinearization.h"

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

namespace {

// A multiplicity-aware complexity score for stride expressions.
// Unlike atoms(), this accounts for repeated symbol use (e.g., s*s or s**2).
size_t stride_complexity_score(const sdfg::symbolic::Expression& expr) {
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return 0;
    }
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        return 1;
    }
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow_expr = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto args = pow_expr->get_args();
        if (args.size() == 2 && SymEngine::is_a<SymEngine::Integer>(*args[1])) {
            try {
                long long exp = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
                if (exp >= 0) {
                    return static_cast<size_t>(exp) * stride_complexity_score(args[0]);
                }
            } catch (const SymEngine::SymEngineException&) {
                return 0;
            }
        }
    }
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        size_t score = 0;
        for (const auto& arg : SymEngine::rcp_static_cast<const SymEngine::Mul>(expr)->get_args()) {
            score += stride_complexity_score(arg);
        }
        return score;
    }
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        size_t score = 0;
        for (const auto& arg : SymEngine::rcp_static_cast<const SymEngine::Add>(expr)->get_args()) {
            score = std::max(score, stride_complexity_score(arg));
        }
        return score;
    }

    // Generic function / composite fallback: recurse through args and add a bonus
    // so function-bearing expressions are preferred over equally-scored plain ones.
    size_t score = 0;
    for (const auto& arg : expr->get_args()) {
        score += stride_complexity_score(arg);
    }
    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        score += 1;
    }
    return score;
}

bool provably_ge(const sdfg::symbolic::Expression& lhs, const sdfg::symbolic::Expression& rhs) {
    return sdfg::symbolic::is_true(sdfg::symbolic::Ge(lhs, rhs));
}

bool provably_gt(const sdfg::symbolic::Expression& lhs, const sdfg::symbolic::Expression& rhs) {
    return provably_ge(lhs, rhs) && !provably_ge(rhs, lhs);
}

} // namespace

DelinearizeResult delinearize(const Expression& expr, const Assumptions& assums) {
    auto dim = expr;

    // Check if more than two symbols are involved
    SymbolVec symbols;
    for (auto& sym : atoms(dim)) {
        if (!assums.at(sym).constant() || !assums.at(sym).map().is_null()) {
            symbols.push_back(sym);
        }
    }
    if (symbols.size() < 2) {
        return {MultiExpression{dim}, MultiExpression{}, true};
    }

    // Step 1: Get polynomial form and affine coefficients
    auto poly = polynomial(dim, symbols);
    if (poly == SymEngine::null) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }
    auto aff_coeffs = affine_coefficients(poly, symbols);
    if (aff_coeffs.empty()) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }
    auto offset = aff_coeffs.at(symbolic::symbol("__daisy_constant__"));
    aff_coeffs.erase(symbolic::symbol("__daisy_constant__"));

    // Step 2: Peel-off dimensions
    DelinearizeResult result;
    MultiExpression strides; // Collect strides, then convert to dimensions
    Expression remaining = symbolic::sub(dim, offset);
    while (!aff_coeffs.empty()) {
        // Pick the symbol with the strongest stride using:
        // 1) provable bound dominance, 2) multiplicity-aware complexity,
        // 3) atom-count fallback.
        Symbol new_dim = SymEngine::null;
        Expression best_coeff = SymEngine::null;
        Expression best_lb = SymEngine::null;
        Expression best_ub = SymEngine::null;
        size_t best_complexity = 0;
        size_t max_atom_count = 0;
        for (const auto& [sym, coeff] : aff_coeffs) {
            auto lb = minimum(coeff, {}, assums);
            auto ub = maximum(coeff, {}, assums);
            size_t complexity = stride_complexity_score(coeff);
            size_t atom_count = symbolic::atoms(coeff).size();

            bool better = false;
            if (new_dim.is_null()) {
                better = true;
            } else {
                // Prefer provably larger lower bound (always positive-stride oriented).
                if (lb != SymEngine::null && best_lb != SymEngine::null && provably_gt(lb, best_lb)) {
                    better = true;
                }
                // If lower bounds are tied/unknown, prefer larger upper bound.
                if (!better && ub != SymEngine::null && best_ub != SymEngine::null && provably_gt(ub, best_ub)) {
                    better = true;
                }
                // Structural fallback that accounts for repeated symbols and pow.
                if (!better && complexity > best_complexity) {
                    better = true;
                }
                // Final deterministic fallback.
                if (!better && complexity == best_complexity && atom_count > max_atom_count) {
                    better = true;
                }
            }

            if (better) {
                max_atom_count = atom_count;
                best_complexity = complexity;
                new_dim = sym;
                best_coeff = coeff;
                best_lb = lb;
                best_ub = ub;
            }
        }
        if (new_dim.is_null()) {
            break;
        }

        // Symbol must be nonnegative
        auto sym_lb = minimum(new_dim, {}, assums);
        if (sym_lb.is_null()) {
            break;
        }
        auto sym_cond = symbolic::Ge(sym_lb, symbolic::zero());
        if (!symbolic::is_true(sym_cond)) {
            break;
        }

        // Stride must be positive
        Expression stride = best_coeff;
        auto stride_lb = best_lb;
        if (stride_lb == SymEngine::null) {
            stride_lb = minimum(stride, {}, assums);
        }
        if (stride_lb.is_null()) {
            break;
        }
        auto stride_cond = symbolic::Ge(stride_lb, symbolic::one());
        if (!symbolic::is_true(stride_cond)) {
            break;
        }

        // Peel off the dimension
        remaining = symbolic::sub(remaining, symbolic::mul(stride, new_dim));
        remaining = symbolic::expand(remaining);
        remaining = symbolic::simplify(remaining);

        // Check if remainder is within bounds

        // remaining must be nonnegative
        auto rem_lb = minimum(remaining, {}, assums);
        if (rem_lb.is_null()) {
            break;
        }
        auto cond_zero = symbolic::Ge(rem_lb, symbolic::zero());
        if (!symbolic::is_true(cond_zero)) {
            break;
        }

        // remaining must be less than stride
        auto ub_stride = (best_ub == SymEngine::null) ? maximum(stride, {}, assums) : best_ub;
        auto ub_remaining = maximum(remaining, {}, assums);
        if (ub_stride == SymEngine::null || ub_remaining == SymEngine::null) {
            break;
        }
        auto cond_stride = symbolic::Ge(ub_stride, ub_remaining);
        if (!symbolic::is_true(cond_stride)) {
            break;
        }

        // Add offset contribution of peeled dimension
        auto [q, r] = polynomial_div(offset, stride);
        offset = r;
        auto final_dim = symbolic::add(new_dim, q);

        result.indices.push_back(final_dim);
        strides.push_back(stride);
        aff_coeffs.erase(new_dim);
    }

    // Not all dimensions could be peeled off
    if (!aff_coeffs.empty()) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }

    // Offset did not reduce to zero
    if (!symbolic::eq(offset, symbolic::zero())) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }

    // Final stride must be 1
    if (!symbolic::eq(strides.back(), symbolic::one())) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }

    // Convert strides to dimensions by dividing consecutive strides
    // For strides [M*K, K, 1], dimensions are [M, K] (M*K/K = M, K/1 = K)
    for (size_t i = 0; i + 1 < strides.size(); ++i) {
        auto [q, r] = polynomial_div(strides[i], strides[i + 1]);
        if (!symbolic::eq(r, symbolic::zero())) {
            // Stride division has non-zero remainder, delinearization failed
            return {MultiExpression{dim}, MultiExpression{}, false};
        }
        result.dimensions.push_back(q);
    }

    result.success = true;
    return result;
}

} // namespace symbolic
} // namespace sdfg
