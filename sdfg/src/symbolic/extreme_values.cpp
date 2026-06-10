#include "sdfg/symbolic/extreme_values.h"

#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/basic.h"
#include "symengine/functions.h"
#include "symengine/number.h"

namespace sdfg {
namespace symbolic {

// ============================================================================
// BoundAnalysis implementation
// ============================================================================

BoundAnalysis::BoundAnalysis(const SymbolSet& parameters, const Assumptions& assumptions, bool use_tight_assumptions)
    : parameters_(parameters), assumptions_(assumptions), use_tight_(use_tight_assumptions) {}

Interval BoundAnalysis::bound(const Expression& expr) { return visit(expr, 0); }

Expression BoundAnalysis::lower_bound(const Expression& expr) { return visit(expr, 0).lower; }

Expression BoundAnalysis::upper_bound(const Expression& expr) { return visit(expr, 0).upper; }

// ============================================================================
// Main dispatch
// ============================================================================

Interval BoundAnalysis::visit(const Expression& expr, size_t depth) {
    if (depth > MAX_DEPTH) {
        return Interval::failure();
    }

    // Cache lookup. The cache stores only fully-successful intervals (both
    // endpoints non-null) AND that were produced without triggering the
    // `visit_symbol` cycle guard anywhere in the subtree. A cycle hit during
    // resolution means the intermediate result depended on the current
    // `visiting_` state — for example, `visit_symbol(j)` called nested under
    // `visit_symbol(i)` may fail to project a coupled constraint because the
    // residue references `i` (which is already in `visiting_`) and returns the
    // un-tightened per-symbol bound. Caching that value would prevent a later
    // top-level call for `j` (with empty `visiting_`) from doing the
    // projection successfully.
    auto it = cache_.find(expr);
    if (it != cache_.end()) {
        return it->second;
    }

    size_t cycle_hits_before = cycle_hits_;
    Interval result = visit_uncached(expr, depth);

    if (result.has_lower() && result.has_upper() && cycle_hits_ == cycle_hits_before) {
        cache_.emplace(expr, result);
    }
    return result;
}

Interval BoundAnalysis::visit_uncached(const Expression& expr, size_t depth) {
    // Fail on NaN / Infty
    if (SymEngine::is_a<SymEngine::NaN>(*expr) || SymEngine::is_a<SymEngine::Infty>(*expr)) {
        return Interval::failure();
    }

    // Integer constant
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return Interval::exact(expr);
    }

    // Rational constant
    if (SymEngine::is_a<SymEngine::Rational>(*expr)) {
        return Interval::exact(expr);
    }

    // Symbol: parameter check (early return before deeper analysis)
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (parameters_.find(sym) != parameters_.end()) {
            return Interval::exact(sym);
        }
        return visit_symbol(sym, depth);
    }

    // Function symbols (idiv, imod, zext_i64, trunc_i32)
    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        return visit_function(SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(expr), depth);
    }

    // Pow
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        return visit_pow(SymEngine::rcp_static_cast<const SymEngine::Pow>(expr), depth);
    }

    // Mul
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        return visit_mul(SymEngine::rcp_static_cast<const SymEngine::Mul>(expr), depth);
    }

    // Add
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        return visit_add(SymEngine::rcp_static_cast<const SymEngine::Add>(expr), depth);
    }

    // Max
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        return visit_max(expr, depth);
    }

    // Min
    if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        return visit_min(expr, depth);
    }

    return Interval::failure();
}

// ============================================================================
// Symbol
// ============================================================================

Interval BoundAnalysis::visit_symbol(const SymEngine::RCP<const SymEngine::Symbol>& sym, size_t depth) {
    // Cycle detection: if we're already computing bounds for this symbol, break the cycle
    if (visiting_.count(sym)) {
        ++cycle_hits_;
        return Interval::failure();
    }
    visiting_.insert(sym);

    auto it = assumptions_.find(sym);
    if (it == assumptions_.end()) {
        visiting_.erase(sym);
        return Interval::failure();
    }
    const auto& assum = it->second;

    Expression lb = SymEngine::null;
    Expression ub = SymEngine::null;

    if (use_tight_) {
        // Tight mode: use tight bounds
        if (!assum.tight_lower_bound().is_null()) {
            lb = visit(assum.tight_lower_bound(), depth + 1).lower;
        }
        if (!assum.tight_upper_bound().is_null()) {
            ub = visit(assum.tight_upper_bound(), depth + 1).upper;
        }
    } else {
        // Loose mode: effective lower bound = max of all lower_bounds
        for (auto& bound : assum.lower_bounds()) {
            auto bound_lb = visit(bound, depth + 1).lower;
            if (bound_lb.is_null()) {
                continue;
            }
            if (lb.is_null()) {
                lb = bound_lb;
            } else {
                lb = symbolic::max(lb, bound_lb);
            }
        }
        // Fall back to tight_lower_bound
        if (lb.is_null() && !assum.tight_lower_bound().is_null()) {
            lb = assum.tight_lower_bound();
        }

        // Effective upper bound = min of all upper_bounds
        for (auto& bound : assum.upper_bounds()) {
            auto bound_ub = visit(bound, depth + 1).upper;
            if (bound_ub.is_null()) {
                continue;
            }
            if (ub.is_null()) {
                ub = bound_ub;
            } else {
                ub = symbolic::min(ub, bound_ub);
            }
        }
        // Fall back to tight_upper_bound
        if (ub.is_null() && !assum.tight_upper_bound().is_null()) {
            ub = assum.tight_upper_bound();
        }
    }

    // Project per-symbol bounds out of coupled affine constraints. Applies
    // in both tight and loose modes — the projection only ever intersects
    // (min for upper, max for lower) with the existing per-symbol bound, so
    // it can only tighten and is sound in either mode.
    //
    // Each constraint `c <= 0` involving `sym` and other symbols can be
    // rearranged via affine inversion into `sym <= U(other syms)` (when
    // the coefficient on `sym` is positive) or `sym >= L(other syms)`
    // (when negative). The residue `U` / `L` is then bounded recursively
    // by the same `BoundAnalysis`; the cycle guard already prevents
    // re-entry on `sym`, so any constraint whose residue ultimately
    // depends on `sym` itself collapses to failure rather than tightening
    // — this is the soundness fence against the "j <= 15-i, i <= 15-j"
    // mutual-dependency case that motivated refusing such literals in
    // `extract_bound_from_literal` historically. The cycle guard also bumps
    // `cycle_hits_`, which prevents `visit()` from caching the un-tightened
    // result so a later top-level query for the same symbol (without the
    // cycle context) can re-run the projection successfully.
    //
    // Sign handling: `solve_affine_bound` only inverts positive
    // coefficients. For negative coefficients, negate the constraint and
    // flip the inequality direction.
    for (const auto& c : assum.constraints()) {
        // Upper-bound projection (positive coeff): solve `c <= 0` for sym
        if (auto u_residue = symbolic::solve_affine_bound(c, sym, symbolic::zero(), /*is_lower_bound=*/false);
            !u_residue.is_null()) {
            auto u_iv = visit(u_residue, depth + 1);
            if (!u_iv.upper.is_null()) {
                ub = ub.is_null() ? u_iv.upper : symbolic::min(ub, u_iv.upper);
            }
        }
        // Lower-bound projection (negative coeff): solve `-c >= 0` for sym
        auto neg_c = symbolic::expand(symbolic::mul(symbolic::integer(-1), c));
        if (auto l_residue = symbolic::solve_affine_bound(neg_c, sym, symbolic::zero(), /*is_lower_bound=*/true);
            !l_residue.is_null()) {
            auto l_iv = visit(l_residue, depth + 1);
            if (!l_iv.lower.is_null()) {
                lb = lb.is_null() ? l_iv.lower : symbolic::max(lb, l_iv.lower);
            }
        }
    }

    // For constant symbols without evolution (outer-scope values, array dimensions),
    // the symbol's value is exactly itself. Use it as fallback when bounds are missing.
    if (assum.constant() && assum.map().is_null()) {
        if (lb.is_null()) lb = sym;
        if (ub.is_null()) ub = sym;
    }

    visiting_.erase(sym);
    return {lb, ub};
}

// ============================================================================
// FunctionSymbol (idiv, imod, zext_i64, trunc_i32)
// ============================================================================

Interval BoundAnalysis::visit_function(const SymEngine::RCP<const SymEngine::FunctionSymbol>& func, size_t depth) {
    auto func_id = func->get_name();

    // zext_i64: monotonic, bounds pass through
    if (func_id == "zext_i64") {
        auto zext = SymEngine::rcp_static_cast<const symbolic::ZExtI64Function>(func);
        auto arg_iv = visit(zext->get_args()[0], depth + 1);
        Expression lb = arg_iv.has_lower() ? symbolic::zext_i64(arg_iv.lower) : SymEngine::null;
        Expression ub = arg_iv.has_upper() ? symbolic::zext_i64(arg_iv.upper) : SymEngine::null;
        return {lb, ub};
    }

    // trunc_i32: monotonic, bounds pass through
    if (func_id == "trunc_i32") {
        auto trunc = SymEngine::rcp_static_cast<const symbolic::TruncI32Function>(func);
        auto arg_iv = visit(trunc->get_args()[0], depth + 1);
        Expression lb = arg_iv.has_lower() ? symbolic::trunc_i32(arg_iv.lower) : SymEngine::null;
        Expression ub = arg_iv.has_upper() ? symbolic::trunc_i32(arg_iv.upper) : SymEngine::null;
        return {lb, ub};
    }

    // idiv(numerator, denominator) — only for constant positive denominator
    if (func_id == "idiv") {
        auto numerator = func->get_args()[0];
        auto denominator = func->get_args()[1];
        if (!SymEngine::is_a<const SymEngine::Integer>(*denominator)) {
            return Interval::failure();
        }

        auto num_iv = visit(numerator, depth + 1);
        auto den_iv = visit(denominator, depth + 1);
        if (!num_iv.has_lower() || !num_iv.has_upper() || !den_iv.has_lower() || !den_iv.has_upper()) {
            return Interval::failure();
        }
        // Denominator must be strictly positive
        if (symbolic::is_true(symbolic::Le(den_iv.lower, symbolic::zero()))) {
            return Interval::failure();
        }
        // For positive denominator: min(a/b) = min(a)/max(b), max(a/b) = max(a)/min(b)
        return {symbolic::div(num_iv.lower, den_iv.upper), symbolic::div(num_iv.upper, den_iv.lower)};
    }

    // imod(lhs, rhs) — only for constant integer rhs
    if (func_id == "imod") {
        auto lhs = func->get_args()[0];
        auto rhs = func->get_args()[1];
        if (!SymEngine::is_a<const SymEngine::Integer>(*rhs)) {
            return Interval::failure();
        }

        auto lhs_iv = visit(lhs, depth + 1);
        if (!lhs_iv.has_lower() || !lhs_iv.has_upper()) {
            return Interval::failure();
        }
        auto lhs_lb = lhs_iv.lower;
        auto lhs_ub = lhs_iv.upper;

        bool can_be_negative = symbolic::is_true(symbolic::Lt(lhs_lb, symbolic::zero())) ||
                               symbolic::is_true(symbolic::Lt(rhs, symbolic::zero()));
        bool all_negative = symbolic::is_true(symbolic::Lt(lhs_ub, symbolic::zero())) ||
                            symbolic::is_true(symbolic::Lt(rhs, symbolic::zero()));
        auto neg_bound = symbolic::sub(symbolic::one(), symbolic::simplify(symbolic::abs(rhs)));
        auto pos_bound = symbolic::sub(rhs, symbolic::one());
        auto zero = symbolic::zero();

        auto width = symbolic::sub(lhs_ub, lhs_lb);
        if (symbolic::is_true(symbolic::Lt(width, rhs))) {
            // Range doesn't span full modulus cycle
            bool wraps = symbolic::is_true(symbolic::Lt(symbolic::mod(lhs_ub, rhs), symbolic::mod(lhs_lb, rhs)));
            if (wraps) {
                Expression lb = can_be_negative ? Expression(neg_bound) : Expression(zero);
                Expression ub = all_negative ? Expression(zero) : Expression(pos_bound);
                return {lb, ub};
            }
            return {symbolic::simplify(symbolic::mod(lhs_lb, rhs)), symbolic::simplify(symbolic::mod(lhs_ub, rhs))};
        }

        // Range spans full cycle
        Expression lb = can_be_negative ? Expression(neg_bound) : Expression(zero);
        Expression ub = all_negative ? Expression(zero) : Expression(pos_bound);
        return {lb, ub};
    }

    return Interval::failure();
}

// ============================================================================
// Pow(base, k) with constant integer exponent k
// ============================================================================

Interval BoundAnalysis::visit_pow(const SymEngine::RCP<const SymEngine::Pow>& pow_expr, size_t depth) {
    auto args = pow_expr->get_args();
    if (args.size() != 2 || !SymEngine::is_a<SymEngine::Integer>(*args[1])) {
        return Interval::failure();
    }

    long long exp_val = 0;
    try {
        exp_val = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
    } catch (const SymEngine::SymEngineException&) {
        return Interval::failure();
    }

    if (exp_val < 0) {
        return Interval::failure();
    }
    if (exp_val == 0) {
        return Interval::exact(symbolic::one());
    }

    auto base_iv = visit(args[0], depth + 1);
    if (!base_iv.has_lower() || !base_iv.has_upper()) {
        return Interval::failure();
    }

    auto exp_expr = symbolic::integer(exp_val);
    auto lb_pow = symbolic::pow(base_iv.lower, exp_expr);
    auto ub_pow = symbolic::pow(base_iv.upper, exp_expr);

    // Odd powers are monotonic
    if (exp_val % 2 != 0) {
        return {lb_pow, ub_pow};
    }

    // Even powers: need sign analysis
    auto zero = symbolic::zero();
    bool interval_nonneg = symbolic::is_true(symbolic::Ge(base_iv.lower, zero));
    bool interval_nonpos = symbolic::is_true(symbolic::Le(base_iv.upper, zero));
    bool crosses_zero = symbolic::is_true(symbolic::Le(base_iv.lower, zero)) &&
                        symbolic::is_true(symbolic::Ge(base_iv.upper, zero));

    if (crosses_zero) {
        // Min is 0, max is the larger of the two endpoint powers
        return {zero, symbolic::max(lb_pow, ub_pow)};
    }
    if (interval_nonneg) {
        // Monotonically increasing
        return {lb_pow, ub_pow};
    }
    if (interval_nonpos) {
        // Monotonically decreasing (for even power on negatives)
        return {ub_pow, lb_pow};
    }

    return Interval::failure();
}

// ============================================================================
// Mul — vertex enumeration of 2^n bound combinations
// ============================================================================

Interval BoundAnalysis::visit_mul(const SymEngine::RCP<const SymEngine::Mul>& mul_expr, size_t depth) {
    const auto& args = mul_expr->get_args();
    size_t n = args.size();

    // Collect intervals for all factors
    std::vector<Interval> factor_ivs;
    factor_ivs.reserve(n);
    bool all_complete = true;

    for (const auto& arg : args) {
        auto iv = visit(arg, depth + 1);
        if (!iv.has_lower() && !iv.has_upper()) {
            return Interval::failure();
        }
        if (!iv.has_lower() || !iv.has_upper()) {
            all_complete = false;
        }
        factor_ivs.push_back(iv);
    }

    // Non-negative fast path: if all factors have non-negative lower bounds,
    // lb = product of lower bounds, ub = product of upper bounds.
    // This avoids symbolic min/max expressions and works with incomplete intervals.
    bool all_nonneg = true;
    for (const auto& iv : factor_ivs) {
        if (!iv.has_lower() || !symbolic::is_true(symbolic::Ge(iv.lower, symbolic::zero()))) {
            all_nonneg = false;
            break;
        }
    }
    if (all_nonneg) {
        Expression lb_product = symbolic::one();
        Expression ub_product = symbolic::one();
        bool ub_valid = true;
        for (const auto& iv : factor_ivs) {
            lb_product = symbolic::mul(lb_product, iv.lower);
            if (iv.has_upper()) {
                ub_product = symbolic::mul(ub_product, iv.upper);
            } else {
                ub_valid = false;
            }
        }
        return {lb_product, ub_valid ? ub_product : SymEngine::null};
    }

    if (!all_complete) {
        return Interval::failure();
    }

    // Enumerate all 2^n vertex combinations
    Expression min_product = SymEngine::null;
    Expression max_product = SymEngine::null;
    const size_t total_combinations = 1ULL << n;

    for (size_t mask = 0; mask < total_combinations; ++mask) {
        Expression product = symbolic::integer(1);
        for (size_t i = 0; i < n; ++i) {
            Expression val = (mask & (1ULL << i)) ? factor_ivs[i].upper : factor_ivs[i].lower;
            product = symbolic::mul(product, val);
        }
        if (min_product.is_null()) {
            min_product = product;
            max_product = product;
        } else {
            min_product = symbolic::min(min_product, product);
            max_product = symbolic::max(max_product, product);
        }
    }

    return {min_product, max_product};
}

// ============================================================================
// Add — polynomial normalization + sign-aware affine bounding
// ============================================================================

Interval BoundAnalysis::visit_add(const SymEngine::RCP<const SymEngine::Add>& add_expr, size_t depth) {
    const auto& args = add_expr->get_args();
    Expression expr = add_expr;

    // Collect generators: non-parameter symbols with a map (loop induction variables).
    // This groups correlated addends (e.g., -2*x + T*x → (T-2)*x) while keeping
    // constant parameters like TSTEPS in the coefficients where they belong.
    SymbolVec gens;
    for (auto& sym : atoms(expr)) {
        if (parameters_.find(sym) != parameters_.end()) {
            continue;
        }
        auto it = assumptions_.find(sym);
        if (it == assumptions_.end()) {
            return Interval::failure();
        }
        // Only include symbols with a map (loop variables).
        // Constant symbols without maps (like TSTEPS) stay in coefficients.
        if (it->second.map().is_null()) {
            continue;
        }
        gens.push_back(sym);
    }

    if (!gens.empty() && gens.size() <= MAX_GENERATORS) {
        auto poly = polynomial(expr, gens);
        if (!poly.is_null()) {
            // Try affine fast-path: if all terms are degree <= 1,
            // use sign-aware monotonicity substitution.
            auto coeffs = affine_coefficients(poly);
            if (!coeffs.empty()) {
                auto result = visit_add_affine(coeffs, gens, depth);

                // Tighten via coupled-constraint projection: per-symbol
                // bounding (inside `visit_add_affine`) sums independent
                // per-generator bounds and so loses the coupling expressed
                // by registered constraints like `x + kx <= N - 1`. The
                // helper looks for a non-negative integer combination of
                // constraints whose generator coefficients match the sum's,
                // yielding a tighter bound on the whole expression.
                if (gens.size() >= 2) {
                    auto coupled = visit_add_coupled_constraints(coeffs, gens, depth);
                    if (coupled.has_upper()) {
                        result.upper = result.has_upper() ? symbolic::min(result.upper, coupled.upper) : coupled.upper;
                    }
                    if (coupled.has_lower()) {
                        result.lower = result.has_lower() ? symbolic::max(result.lower, coupled.lower) : coupled.lower;
                    }
                }

                if (result.has_lower() || result.has_upper()) {
                    return result;
                }
            }

            // General polynomial: bound each monomial term independently
            auto result = visit_add_polynomial(poly, gens, depth);
            if (result.has_lower() || result.has_upper()) {
                return result;
            }
            // Fall through to arg-wise if polynomial bounding failed
        }
    }

    // Fallback: arg-wise bounding (sound for independent addends)
    return visit_add_argwise(args, depth);
}

// ============================================================================
// Max / Min
// ============================================================================

Interval BoundAnalysis::visit_max(const Expression& expr, size_t depth) {
    auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Max>(expr)->get_args();

    Expression lb = SymEngine::null;
    Expression ub = SymEngine::null;

    for (const auto& arg : args) {
        auto iv = visit(arg, depth + 1);
        if (!iv.has_lower() || !iv.has_upper()) {
            return Interval::failure();
        }

        // Lower bound of max(a,b,...) = max(lb_a, lb_b, ...)
        lb = lb.is_null() ? iv.lower : symbolic::max(lb, iv.lower);

        // Upper bound of max(a,b,...) = max(ub_a, ub_b, ...)
        ub = ub.is_null() ? iv.upper : symbolic::max(ub, iv.upper);
    }

    return {lb, ub};
}

Interval BoundAnalysis::visit_min(const Expression& expr, size_t depth) {
    auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Min>(expr)->get_args();

    Expression lb = SymEngine::null;
    Expression ub = SymEngine::null;

    for (const auto& arg : args) {
        auto iv = visit(arg, depth + 1);
        if (!iv.has_lower() || !iv.has_upper()) {
            return Interval::failure();
        }

        // Lower bound of min(a,b,...) = min(lb_a, lb_b, ...)
        lb = lb.is_null() ? iv.lower : symbolic::min(lb, iv.lower);

        // Upper bound of min(a,b,...) = min(ub_a, ub_b, ...)
        ub = ub.is_null() ? iv.upper : symbolic::min(ub, iv.upper);
    }

    return {lb, ub};
}

// ============================================================================
// Add helpers
// ============================================================================

Interval BoundAnalysis::visit_add_affine(const AffineCoeffs& coeffs, const SymbolVec& gens, size_t depth) {
    Expression lb_sum = SymEngine::null;
    Expression ub_sum = SymEngine::null;
    auto constant_sym = symbolic::symbol("__daisy_constant__");
    bool lb_valid = true;
    bool ub_valid = true;

    for (auto& [sym, coeff] : coeffs) {
        // Handle constant term
        if (symbolic::eq(sym, constant_sym)) {
            auto coeff_iv = visit(coeff, depth + 1);
            if (coeff_iv.has_lower() && lb_valid) {
                lb_sum = lb_sum.is_null() ? coeff_iv.lower : symbolic::add(lb_sum, coeff_iv.lower);
            } else {
                lb_valid = false;
            }
            if (coeff_iv.has_upper() && ub_valid) {
                ub_sum = ub_sum.is_null() ? coeff_iv.upper : symbolic::add(ub_sum, coeff_iv.upper);
            } else {
                ub_valid = false;
            }
            continue;
        }

        // For term coeff * sym: use sign of coefficient to determine monotonicity
        auto sym_iv = visit(sym, depth + 1);
        auto coeff_iv = visit(coeff, depth + 1);

        // Determine sign of coefficient (only needs one bound direction)
        bool coeff_nonneg = coeff_iv.has_lower() && symbolic::is_true(symbolic::Ge(coeff_iv.lower, symbolic::zero()));
        bool coeff_nonpos = coeff_iv.has_upper() && symbolic::is_true(symbolic::Le(coeff_iv.upper, symbolic::zero()));

        Expression term_lb = SymEngine::null;
        Expression term_ub = SymEngine::null;

        if (coeff_nonneg) {
            // Monotonically increasing in sym: lb = coeff_lb * sym_lb, ub = coeff_ub * sym_ub
            if (coeff_iv.has_lower() && sym_iv.has_lower()) {
                term_lb = symbolic::mul(coeff_iv.lower, sym_iv.lower);
            }
            if (coeff_iv.has_upper() && sym_iv.has_upper()) {
                term_ub = symbolic::mul(coeff_iv.upper, sym_iv.upper);
            }
        } else if (coeff_nonpos) {
            // Monotonically decreasing in sym: lb = coeff_lb * sym_ub, ub = coeff_ub * sym_lb
            if (coeff_iv.has_lower() && sym_iv.has_upper()) {
                term_lb = symbolic::mul(coeff_iv.lower, sym_iv.upper);
            }
            if (coeff_iv.has_upper() && sym_iv.has_lower()) {
                term_ub = symbolic::mul(coeff_iv.upper, sym_iv.lower);
            }
        } else {
            // Unknown sign: need all 4 bounds for corner enumeration
            if (coeff_iv.has_lower() && coeff_iv.has_upper() && sym_iv.has_lower() && sym_iv.has_upper()) {
                auto a = symbolic::mul(coeff_iv.lower, sym_iv.lower);
                auto b = symbolic::mul(coeff_iv.lower, sym_iv.upper);
                auto c = symbolic::mul(coeff_iv.upper, sym_iv.lower);
                auto d = symbolic::mul(coeff_iv.upper, sym_iv.upper);
                term_lb = symbolic::min(symbolic::min(a, b), symbolic::min(c, d));
                term_ub = symbolic::max(symbolic::max(a, b), symbolic::max(c, d));
            }
        }

        if (!term_lb.is_null() && lb_valid) {
            lb_sum = lb_sum.is_null() ? term_lb : symbolic::add(lb_sum, term_lb);
        } else {
            lb_valid = false;
        }
        if (!term_ub.is_null() && ub_valid) {
            ub_sum = ub_sum.is_null() ? term_ub : symbolic::add(ub_sum, term_ub);
        } else {
            ub_valid = false;
        }
    }

    return {lb_valid ? lb_sum : SymEngine::null, ub_valid ? ub_sum : SymEngine::null};
}

Interval BoundAnalysis::visit_add_polynomial(const Polynomial& poly, const SymbolVec& gens, size_t depth) {
    auto& D = poly->get_poly().get_dict();
    Expression lb_sum = SymEngine::null;
    Expression ub_sum = SymEngine::null;
    bool lb_valid = true;
    bool ub_valid = true;

    for (auto& [exponents, coeff] : D) {
        // Reconstruct term: coeff * gen_0^e0 * gen_1^e1 * ...
        Expression term = coeff;
        for (size_t i = 0; i < exponents.size(); i++) {
            if (exponents[i] != 0) {
                term = symbolic::mul(term, symbolic::pow(gens[i], symbolic::integer(exponents[i])));
            }
        }
        auto term_iv = visit(term, depth + 1);
        if (term_iv.has_lower() && lb_valid) {
            lb_sum = lb_sum.is_null() ? term_iv.lower : symbolic::add(lb_sum, term_iv.lower);
        } else {
            lb_valid = false;
        }
        if (term_iv.has_upper() && ub_valid) {
            ub_sum = ub_sum.is_null() ? term_iv.upper : symbolic::add(ub_sum, term_iv.upper);
        } else {
            ub_valid = false;
        }
    }

    return {lb_valid ? lb_sum : SymEngine::null, ub_valid ? ub_sum : SymEngine::null};
}

Interval BoundAnalysis::visit_add_argwise(const SymEngine::vec_basic& args, size_t depth) {
    Expression lb_sum = SymEngine::null;
    Expression ub_sum = SymEngine::null;
    bool lb_valid = true;
    bool ub_valid = true;

    for (const auto& arg : args) {
        auto iv = visit(arg, depth + 1);
        if (iv.has_lower() && lb_valid) {
            lb_sum = lb_sum.is_null() ? iv.lower : symbolic::add(lb_sum, iv.lower);
        } else {
            lb_valid = false;
        }
        if (iv.has_upper() && ub_valid) {
            ub_sum = ub_sum.is_null() ? iv.upper : symbolic::add(ub_sum, iv.upper);
        } else {
            ub_valid = false;
        }
    }

    return {lb_valid ? lb_sum : SymEngine::null, ub_valid ? ub_sum : SymEngine::null};
}

// ============================================================================
// Coupled-constraint projection helper for affine sums
// ============================================================================

namespace {

// Extract the integer coefficient of `gen` from an `AffineCoeffs` map.
// Returns true on success. Non-integer coefficients are rejected (the
// helper deliberately stays in the integer regime).
bool integer_coeff(const AffineCoeffs& coeffs, const Symbol& gen, long long& out) {
    auto it = coeffs.find(gen);
    if (it == coeffs.end()) {
        out = 0;
        return true;
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*it->second)) {
        return false;
    }
    out = SymEngine::rcp_static_cast<const SymEngine::Integer>(it->second)->as_int();
    return true;
}

// Convert a registered constraint expression into its affine decomposition
// over the given generator vector. Returns false if the constraint is not
// purely affine with integer generator coefficients (the helper only
// reasons in the integer regime).
//
// The constant-part expression is returned via `out_const` and may still
// contain parameters (constant assumption symbols, array dimensions, ...).
bool decompose_constraint(
    const Expression& constraint, const SymbolVec& gens, std::vector<long long>& out_g_coeffs, Expression& out_const
) {
    SymbolVec gens_copy = gens;
    auto poly = polynomial(constraint, gens_copy);
    if (poly.is_null()) return false;
    auto coeffs = affine_coefficients(poly);
    if (coeffs.empty()) return false;

    out_g_coeffs.assign(gens.size(), 0);
    out_const = symbolic::zero();

    auto constant_sym = symbolic::symbol("__daisy_constant__");
    for (const auto& [sym, coeff] : coeffs) {
        if (symbolic::eq(sym, constant_sym)) {
            out_const = coeff;
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) return false;
        long long v = SymEngine::rcp_static_cast<const SymEngine::Integer>(coeff)->as_int();
        if (v == 0) continue;
        // Locate this symbol in `gens`.
        bool found = false;
        for (size_t i = 0; i < gens.size(); ++i) {
            if (symbolic::eq(gens[i], sym)) {
                out_g_coeffs[i] = v;
                found = true;
                break;
            }
        }
        if (!found) {
            // Constraint mentions a generator outside the target sum's
            // generator set. We can't cancel that contribution, so this
            // constraint cannot participate.
            return false;
        }
    }
    return true;
}

} // namespace

Interval BoundAnalysis::visit_add_coupled_constraints(const AffineCoeffs& coeffs, const SymbolVec& gens, size_t depth) {
    if (gens.empty()) return Interval::failure();

    auto constant_sym = symbolic::symbol("__daisy_constant__");

    // Materialize the sum's generator coefficients and constant part.
    std::vector<long long> sum_g(gens.size(), 0);
    Expression sum_const = symbolic::zero();
    for (size_t i = 0; i < gens.size(); ++i) {
        long long c = 0;
        if (!integer_coeff(coeffs, gens[i], c)) return Interval::failure();
        sum_g[i] = c;
    }
    {
        auto it = coeffs.find(constant_sym);
        if (it != coeffs.end()) sum_const = it->second;
    }

    // Collect candidate constraints from every generator's assumption,
    // deduplicating by SymEngine equality.
    struct Candidate {
        Expression expr;
        std::vector<long long> g_coeffs;
        Expression const_part;
    };
    std::vector<Candidate> cands;
    ExpressionSet seen;
    for (const auto& g : gens) {
        auto a_it = assumptions_.find(g);
        if (a_it == assumptions_.end()) continue;
        for (const auto& c : a_it->second.constraints()) {
            if (!seen.insert(c).second) continue;
            Candidate cand;
            cand.expr = c;
            if (!decompose_constraint(c, gens, cand.g_coeffs, cand.const_part)) continue;
            cands.push_back(std::move(cand));
        }
    }
    if (cands.empty()) return Interval::failure();

    // Greedy peel in one direction, with partial-peel fallback. `direction
    // = +1` projects toward an upper bound (each constraint `c <= 0` is
    // scaled by a non-negative lambda); `direction = -1` projects toward a
    // lower bound (using `-c >= 0`, scaled by non-negative mu). The same
    // loop computes both directions by negating the residual sign.
    //
    // After the loop, ANY leftover residual is bounded per-symbol and added
    // back: `expr <= K_now + sum(residual_now[i] * gens[i])` (for
    // direction=+1). This recovers the partial benefit even when the sum
    // has free terms (`c` in im2col) that no registered constraint mentions.
    // If residual fully cancels, the per-symbol fallback contributes 0 and
    // the returned bound is just `K_now`.
    //
    // Returns an expression `E` such that `direction * expr <= E`. For
    // direction=+1 the caller uses `E.upper` as `expr`'s upper bound; for
    // direction=-1 it negates `E` and uses `(-E).lower` as `expr`'s lower
    // bound.
    auto try_direction = [&](int direction) -> Expression {
        std::vector<long long> residual = sum_g;
        for (auto& v : residual) v *= direction;
        Expression accumulated_const = sum_const;
        if (direction < 0) accumulated_const = symbolic::mul(symbolic::integer(-1), accumulated_const);

        for (int iter = 0; iter < 32; ++iter) {
            // Walk all generators; apply the first (generator, constraint)
            // pair that makes non-overshooting progress. This is critical
            // when the sum mixes constraint-capable terms (`hout`, `kh`)
            // with free terms (`c`) - picking the first non-zero residual
            // would otherwise bail immediately on the free term.
            bool advanced = false;
            for (size_t pick = 0; pick < gens.size() && !advanced; ++pick) {
                if (residual[pick] == 0) continue;
                long long need = residual[pick];
                for (const auto& c : cands) {
                    long long cc = c.g_coeffs[pick];
                    if (cc == 0) continue;
                    if (need % cc != 0) continue;
                    long long lambda = need / cc;
                    if (lambda <= 0) continue;

                    // Tentatively subtract `lambda * c` from the residual.
                    std::vector<long long> r_new = residual;
                    bool overshoot = false;
                    for (size_t i = 0; i < gens.size(); ++i) {
                        long long sub = lambda * c.g_coeffs[i];
                        long long old_val = r_new[i];
                        long long new_val = old_val - sub;
                        // Don't allow a coefficient to grow in absolute value:
                        // greedy must always move toward zero.
                        if (old_val == 0) {
                            if (new_val != 0) {
                                overshoot = true;
                                break;
                            }
                        } else {
                            long long abs_old = old_val < 0 ? -old_val : old_val;
                            long long abs_new = new_val < 0 ? -new_val : new_val;
                            if (abs_new > abs_old) {
                                overshoot = true;
                                break;
                            }
                        }
                        r_new[i] = new_val;
                    }
                    if (overshoot) continue;

                    residual = std::move(r_new);
                    accumulated_const =
                        symbolic::sub(accumulated_const, symbolic::mul(symbolic::integer(lambda), c.const_part));
                    advanced = true;
                    break;
                }
            }
            if (!advanced) break;
        }

        // Build the partial bound: `K + sum(residual[i] * gens[i])`. For
        // each leftover term, substitute the generator with its per-symbol
        // upper/lower bound (sign of `residual[i]` picks which). Using
        // `visit_symbol` (NOT `visit`) avoids re-entering `visit_add` and
        // therefore this helper, keeping the recursion bounded.
        Expression bound = accumulated_const;
        for (size_t i = 0; i < gens.size(); ++i) {
            if (residual[i] == 0) continue;
            auto sym_iv = visit(gens[i], depth + 1);
            Expression chosen;
            if (residual[i] > 0) {
                if (!sym_iv.has_upper()) return Expression(SymEngine::null);
                chosen = sym_iv.upper;
            } else {
                if (!sym_iv.has_lower()) return Expression(SymEngine::null);
                chosen = sym_iv.lower;
            }
            bound = symbolic::add(bound, symbolic::mul(symbolic::integer(residual[i]), chosen));
        }
        return bound;
    };

    Expression upper_k = try_direction(+1);
    Expression lower_k = try_direction(-1);

    Expression ub = SymEngine::null;
    if (!upper_k.is_null()) {
        // `K` may still mention parameters - resolve via the regular flow.
        auto k_iv = visit(symbolic::expand(upper_k), depth + 1);
        if (k_iv.has_upper()) ub = k_iv.upper;
    }
    Expression lb = SymEngine::null;
    if (!lower_k.is_null()) {
        // For the lower bound: `-residual = expr - sum(mu_i * c_i)`, so
        // `expr = -K_lower + sum(mu_i * (-c_i)) >= -K_lower`. Negate.
        auto neg_k = symbolic::mul(symbolic::integer(-1), lower_k);
        auto k_iv = visit(symbolic::expand(neg_k), depth + 1);
        if (k_iv.has_lower()) lb = k_iv.lower;
    }

    return {lb, ub};
}

// ============================================================================
// Backward-compatible free functions
// ============================================================================

Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight) {
    BoundAnalysis analysis(parameters, assumptions, tight);
    return analysis.lower_bound(expr);
}

Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight) {
    BoundAnalysis analysis(parameters, assumptions, tight);
    return analysis.upper_bound(expr);
}

// ============================================================================
// Inequality proofs
// ============================================================================
//
// All `is_*` predicates are sound and incomplete: TRUE means proven; FALSE
// means unknown OR disproven. Implementation chain:
//   1. Direct boolean check on the predicate operator.
//   2. BoundAnalysis interval check on `(a - b)` against zero.
//   3. Min/Max descent: substitute one Min/Max subexpression with each of its
//      arguments (one at a time) and recurse, since
//          Max(a_1,...,a_n) >= a_i  and  Min(a_1,...,a_n) <= a_i.
//      Substituting Max with a smaller arg yields a LOWER bound on the
//      enclosing expression; substituting Min with any arg yields an UPPER
//      bound. Both are sound when the Min/Max appears in a monotone-
//      nondecreasing position (e.g. additive top-level), which is the case
//      for the affine subscript expressions this API targets.

namespace {

// Find the first Min subexpression (DFS). Returns null if none.
SymEngine::RCP<const SymEngine::Basic> find_first_min(const Expression& expr) {
    std::vector<SymEngine::RCP<const SymEngine::Basic>> worklist;
    worklist.push_back(expr);
    while (!worklist.empty()) {
        auto node = worklist.back();
        worklist.pop_back();
        if (SymEngine::is_a<SymEngine::Min>(*node)) return node;
        for (auto& a : node->get_args()) worklist.push_back(a);
    }
    return SymEngine::null;
}

// Find the first Max subexpression (DFS). Returns null if none.
SymEngine::RCP<const SymEngine::Basic> find_first_max(const Expression& expr) {
    std::vector<SymEngine::RCP<const SymEngine::Basic>> worklist;
    worklist.push_back(expr);
    while (!worklist.empty()) {
        auto node = worklist.back();
        worklist.pop_back();
        if (SymEngine::is_a<SymEngine::Max>(*node)) return node;
        for (auto& a : node->get_args()) worklist.push_back(a);
    }
    return SymEngine::null;
}

constexpr int kProofDepthLimit = 4;

// Forward decl: shared core that proves `diff >= 0` (when strict=false)
// or `diff > 0` (when strict=true).
bool prove_ge_zero(
    const Expression& diff,
    const SymbolSet& parameters,
    const Assumptions& assumptions,
    bool tight,
    bool strict,
    int depth
);

// Substitute the first Max subexpression with one of its args at a time, and
// recurse: any successful branch proves the predicate for the original expr
// because the substitution yields a sound LOWER bound on the residue.
bool descend_max(
    const Expression& diff,
    const SymbolSet& parameters,
    const Assumptions& assumptions,
    bool tight,
    bool strict,
    int depth
) {
    auto max_node = find_first_max(diff);
    if (max_node.is_null()) return false;
    auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(max_node);
    for (auto& arg : max_op->get_args()) {
        Expression replaced = symbolic::simplify(symbolic::expand(symbolic::subs(diff, max_node, arg)));
        if (prove_ge_zero(replaced, parameters, assumptions, tight, strict, depth - 1)) return true;
    }
    return false;
}

// AND-style Min descent: when proving `e >= 0` (or `> 0`) and `e` contains a
// `Min` subexpression in monotone-nondecreasing position, the equality
// `f(min(a, b)) = min(f(a), f(b))` (for f nondecreasing) lets us replace Min
// with each arg in turn and require ALL substitutions to be provable.
//
// Conservative monotonicity check: the original `e` must be either the Min
// itself, or an Add containing the Min as a direct addend (so the implicit
// coefficient is +1). Anything else is rejected to avoid unsound descent
// through negations or non-positive multipliers.
bool descend_min_and(
    const Expression& e, const SymbolSet& parameters, const Assumptions& assumptions, bool tight, bool strict, int depth
) {
    auto min_node = find_first_min(e);
    if (min_node.is_null()) return false;

    // Monotonicity check: Min must appear at top level or as a direct addend.
    bool ok = false;
    if (e.get() == min_node.get()) {
        ok = true;
    } else if (SymEngine::is_a<SymEngine::Add>(*e)) {
        for (const auto& term : e->get_args()) {
            if (term.get() == min_node.get() || symbolic::eq(term, min_node)) {
                ok = true;
                break;
            }
        }
    }
    if (!ok) return false;

    auto min_op = SymEngine::rcp_static_cast<const SymEngine::Min>(min_node);
    for (auto& arg : min_op->get_args()) {
        Expression replaced = symbolic::simplify(symbolic::expand(symbolic::subs(e, min_node, arg)));
        if (!prove_ge_zero(replaced, parameters, assumptions, tight, strict, depth - 1)) return false;
    }
    return true;
}

bool prove_ge_zero(
    const Expression& diff,
    const SymbolSet& parameters,
    const Assumptions& assumptions,
    bool tight,
    bool strict,
    int depth
) {
    auto e = symbolic::expand(diff);

    // Constant integer fast path.
    if (SymEngine::is_a<SymEngine::Integer>(*e)) {
        auto i = SymEngine::rcp_static_cast<const SymEngine::Integer>(e);
        return strict ? i->is_positive() : !i->is_negative();
    }

    // Direct decidable check.
    if (strict) {
        if (symbolic::is_true(symbolic::Gt(e, symbolic::zero()))) return true;
    } else {
        if (symbolic::is_true(symbolic::Ge(e, symbolic::zero()))) return true;
    }

    // Interval check via BoundAnalysis with the supplied parameter set.
    auto try_lb = [&](const SymbolSet& params) -> bool {
        BoundAnalysis analysis(params, assumptions, tight);
        auto lb = analysis.lower_bound(e);
        if (lb.is_null() || SymEngine::is_a<SymEngine::Infty>(*lb)) return false;
        // Simplify the computed bound: BoundAnalysis can return shapes like
        // `N - (N - 1)` that don't reduce inside `is_true(Ge(...))`.
        auto lb_s = symbolic::simplify(symbolic::expand(lb));
        if (SymEngine::is_a<SymEngine::Integer>(*lb_s)) {
            auto i = SymEngine::rcp_static_cast<const SymEngine::Integer>(lb_s);
            if (strict ? i->is_positive() : !i->is_negative()) return true;
        }
        if (strict) {
            if (symbolic::is_true(symbolic::Gt(lb_s, symbolic::zero()))) return true;
        } else {
            if (symbolic::is_true(symbolic::Ge(lb_s, symbolic::zero()))) return true;
        }
        // Max-descent on the computed lower bound: tight bounds frequently
        // take the shape `c + max(0, X)`. Substituting Max with one arg yields
        // a (sound) lower bound on `lb`, which transitively bounds `e`.
        if (depth > 0 && descend_max(lb_s, params, assumptions, tight, strict, depth - 1)) return true;
        // Min-descent (AND): `BoundAnalysis` may emit shapes like
        // `N + min(0, 1 - N)` whose value depends on the Min branches.
        // For each Min arg, substitute and require ALL branches to be
        // provable (sound when Min sits in monotone-nondecreasing position).
        if (depth > 0 && descend_min_and(lb_s, params, assumptions, tight, strict, depth - 1)) return true;
        return false;
    };
    // First with the caller's parameters (preserves chain-resolution shapes
    // like `upper(i) = N - 1` when N is a parameter).
    if (try_lb(parameters)) return true;
    // Fallback with empty parameters: lets BoundAnalysis substitute
    // assumption-derived bounds on parameters themselves (e.g. `N >= 1`).
    if (!parameters.empty() && try_lb({})) return true;

    if (depth <= 0) return false;

    // Max descent on the original expression.
    if (descend_max(e, parameters, assumptions, tight, strict, depth)) return true;

    return false;
}

} // namespace

bool is_nonneg(const Expression& expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight) {
    return prove_ge_zero(expr, parameters, assumptions, tight, /*strict=*/false, kProofDepthLimit);
}

bool is_positive(const Expression& expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight) {
    return prove_ge_zero(expr, parameters, assumptions, tight, /*strict=*/true, kProofDepthLimit);
}

bool is_nonpos(const Expression& expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight) {
    // expr <= 0  iff  -expr >= 0
    return is_nonneg(symbolic::mul(symbolic::integer(-1), expr), parameters, assumptions, tight);
}

bool is_negative(const Expression& expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight) {
    // expr < 0  iff  -expr > 0
    return is_positive(symbolic::mul(symbolic::integer(-1), expr), parameters, assumptions, tight);
}

bool is_ge(
    const Expression& a, const Expression& b, const SymbolSet& parameters, const Assumptions& assumptions, bool tight
) {
    return is_nonneg(symbolic::sub(a, b), parameters, assumptions, tight);
}

bool is_gt(
    const Expression& a, const Expression& b, const SymbolSet& parameters, const Assumptions& assumptions, bool tight
) {
    // For `a > b`, descending Min on the LHS is also sound: `a > min(x,y)` if
    // `a > x` OR `a > y`. Implement by routing through `is_positive(a - b)`
    // then, if that fails, doing a Min descent on `b` (i.e. on the negative
    // term of the difference).
    auto diff = symbolic::sub(a, b);
    if (is_positive(diff, parameters, assumptions, tight)) return true;
    // Min descent on b: a > min(args) iff a > arg_i for some i.
    auto min_node = find_first_min(b);
    if (!min_node.is_null()) {
        auto min_op = SymEngine::rcp_static_cast<const SymEngine::Min>(min_node);
        for (auto& arg : min_op->get_args()) {
            Expression b_replaced = symbolic::simplify(symbolic::expand(symbolic::subs(b, min_node, arg)));
            if (is_gt(a, b_replaced, parameters, assumptions, tight)) return true;
        }
    }
    return false;
}

bool is_le(
    const Expression& a, const Expression& b, const SymbolSet& parameters, const Assumptions& assumptions, bool tight
) {
    return is_ge(b, a, parameters, assumptions, tight);
}

bool is_lt(
    const Expression& a, const Expression& b, const SymbolSet& parameters, const Assumptions& assumptions, bool tight
) {
    return is_gt(b, a, parameters, assumptions, tight);
}

bool is_eq(
    const Expression& a, const Expression& b, const SymbolSet& parameters, const Assumptions& assumptions, bool tight
) {
    if (symbolic::eq(a, b)) return true;
    return is_ge(a, b, parameters, assumptions, tight) && is_ge(b, a, parameters, assumptions, tight);
}

// ============================================================================
// BoundAnalysis-based overloads (cache-amortized)
// ============================================================================
//
// These mirror the legacy `is_*` predicates but route every interval query
// through the caller-supplied `BoundAnalysis`. Semantics match the legacy
// versions when invoked with a `BoundAnalysis` constructed from the same
// `(parameters, assumptions, tight)`; the only difference is that the
// empty-parameter fallback (`try_lb({})`) is omitted, since callers that
// want it can construct a second `BoundAnalysis` and call the predicate
// twice.

namespace {

bool prove_ge_zero_ba(const Expression& diff, BoundAnalysis& ba, bool strict, int depth);

bool descend_max_ba(const Expression& diff, BoundAnalysis& ba, bool strict, int depth) {
    auto max_node = find_first_max(diff);
    if (max_node.is_null()) return false;
    auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(max_node);
    for (auto& arg : max_op->get_args()) {
        Expression replaced = symbolic::simplify(symbolic::expand(symbolic::subs(diff, max_node, arg)));
        if (prove_ge_zero_ba(replaced, ba, strict, depth - 1)) return true;
    }
    return false;
}

bool descend_min_and_ba(const Expression& e, BoundAnalysis& ba, bool strict, int depth) {
    auto min_node = find_first_min(e);
    if (min_node.is_null()) return false;

    bool ok = false;
    if (e.get() == min_node.get()) {
        ok = true;
    } else if (SymEngine::is_a<SymEngine::Add>(*e)) {
        for (const auto& term : e->get_args()) {
            if (term.get() == min_node.get() || symbolic::eq(term, min_node)) {
                ok = true;
                break;
            }
        }
    }
    if (!ok) return false;

    auto min_op = SymEngine::rcp_static_cast<const SymEngine::Min>(min_node);
    for (auto& arg : min_op->get_args()) {
        Expression replaced = symbolic::simplify(symbolic::expand(symbolic::subs(e, min_node, arg)));
        if (!prove_ge_zero_ba(replaced, ba, strict, depth - 1)) return false;
    }
    return true;
}

bool prove_ge_zero_ba(const Expression& diff, BoundAnalysis& ba, bool strict, int depth) {
    auto e = symbolic::expand(diff);

    // Constant integer fast path.
    if (SymEngine::is_a<SymEngine::Integer>(*e)) {
        auto i = SymEngine::rcp_static_cast<const SymEngine::Integer>(e);
        return strict ? i->is_positive() : !i->is_negative();
    }

    // Direct decidable check.
    if (strict) {
        if (symbolic::is_true(symbolic::Gt(e, symbolic::zero()))) return true;
    } else {
        if (symbolic::is_true(symbolic::Ge(e, symbolic::zero()))) return true;
    }

    // Interval check via the supplied BoundAnalysis.
    auto lb = ba.lower_bound(e);
    if (!lb.is_null() && !SymEngine::is_a<SymEngine::Infty>(*lb)) {
        auto lb_s = symbolic::simplify(symbolic::expand(lb));
        if (SymEngine::is_a<SymEngine::Integer>(*lb_s)) {
            auto i = SymEngine::rcp_static_cast<const SymEngine::Integer>(lb_s);
            if (strict ? i->is_positive() : !i->is_negative()) return true;
        }
        if (strict) {
            if (symbolic::is_true(symbolic::Gt(lb_s, symbolic::zero()))) return true;
        } else {
            if (symbolic::is_true(symbolic::Ge(lb_s, symbolic::zero()))) return true;
        }
        if (depth > 0 && descend_max_ba(lb_s, ba, strict, depth - 1)) return true;
        if (depth > 0 && descend_min_and_ba(lb_s, ba, strict, depth - 1)) return true;
    }

    if (depth <= 0) return false;

    // Max descent on the original expression.
    if (descend_max_ba(e, ba, strict, depth)) return true;

    return false;
}

} // namespace

bool is_nonneg(const Expression& expr, BoundAnalysis& ba) {
    return prove_ge_zero_ba(expr, ba, /*strict=*/false, kProofDepthLimit);
}

bool is_positive(const Expression& expr, BoundAnalysis& ba) {
    return prove_ge_zero_ba(expr, ba, /*strict=*/true, kProofDepthLimit);
}

bool is_nonpos(const Expression& expr, BoundAnalysis& ba) {
    return is_nonneg(symbolic::mul(symbolic::integer(-1), expr), ba);
}

bool is_negative(const Expression& expr, BoundAnalysis& ba) {
    return is_positive(symbolic::mul(symbolic::integer(-1), expr), ba);
}

bool is_ge(const Expression& a, const Expression& b, BoundAnalysis& ba) { return is_nonneg(symbolic::sub(a, b), ba); }

bool is_gt(const Expression& a, const Expression& b, BoundAnalysis& ba) {
    auto diff = symbolic::sub(a, b);
    if (is_positive(diff, ba)) return true;
    // Min descent on b: a > min(args) iff a > arg_i for some i.
    auto min_node = find_first_min(b);
    if (!min_node.is_null()) {
        auto min_op = SymEngine::rcp_static_cast<const SymEngine::Min>(min_node);
        for (auto& arg : min_op->get_args()) {
            Expression b_replaced = symbolic::simplify(symbolic::expand(symbolic::subs(b, min_node, arg)));
            if (is_gt(a, b_replaced, ba)) return true;
        }
    }
    return false;
}

bool is_le(const Expression& a, const Expression& b, BoundAnalysis& ba) { return is_ge(b, a, ba); }

bool is_lt(const Expression& a, const Expression& b, BoundAnalysis& ba) { return is_gt(b, a, ba); }

bool is_eq(const Expression& a, const Expression& b, BoundAnalysis& ba) {
    if (symbolic::eq(a, b)) return true;
    return is_ge(a, b, ba) && is_ge(b, a, ba);
}

} // namespace symbolic
} // namespace sdfg
