#include "sdfg/symbolic/polynomials.h"

#include <symengine/polys/basic_conversions.h>

namespace sdfg {
namespace symbolic {

Polynomial polynomial(const Expression expr, SymbolVec& symbols) {
    try {
        ExpressionSet gens;
        for (auto& symbol : symbols) {
            gens.insert(symbol);
        }
        return SymEngine::from_basic<SymEngine::MExprPoly>(expr, gens);
    } catch (SymEngine::SymEngineException& e) {
        return SymEngine::null;
    }
};

AffineCoeffs affine_coefficients(Polynomial poly, SymbolVec& symbols) {
    AffineCoeffs coeffs;
    for (auto& symbol : symbols) {
        coeffs[symbol] = symbolic::zero();
    }
    coeffs[symbolic::symbol("__daisy_constant__")] = symbolic::zero();

    auto& D = poly->get_poly().get_dict();
    for (auto& [exponents, coeff] : D) {
        // Check if sum of exponents is <= 1
        symbolic::Symbol symbol = symbolic::symbol("__daisy_constant__");
        unsigned total_deg = 0;
        for (size_t i = 0; i < exponents.size(); i++) {
            auto& e = exponents[i];
            if (e > 0) {
                symbol = symbols[i];
            }
            total_deg += e;
        }
        if (total_deg > 1) {
            return {};
        }

        // Add coefficient to corresponding symbol
        coeffs[symbol] = symbolic::add(coeffs[symbol], coeff);
    }

    return coeffs;
}

Expression affine_inverse(AffineCoeffs coeffs, Symbol symbol) {
    if (!coeffs.contains(symbol) || eq(coeffs[symbol], zero())) {
        return SymEngine::null;
    }

    Expression result = symbol;
    for (auto& [sym, expr] : coeffs) {
        if (eq(sym, symbol)) {
            continue;
        }
        result = symbolic::add(result, SymEngine::neg(expr));
    }

    return symbolic::div(result, coeffs[symbol]);
}

std::pair<std::vector<int>, Expression> get_leading_term(const Polynomial& poly) {
    if (poly->get_poly().dict_.empty()) {
        return {{}, symbolic::zero()};
    }

    auto it = poly->get_poly().dict_.begin();
    std::vector<int> max_exp = it->first;
    Expression max_coeff = it->second;

    for (++it; it != poly->get_poly().dict_.end(); ++it) {
        // Compare exponents lexicographically
        bool greater = false;
        for (size_t i = 0; i < max_exp.size(); ++i) {
            if (it->first[i] > max_exp[i]) {
                greater = true;
                break;
            } else if (it->first[i] < max_exp[i]) {
                break;
            }
        }
        if (greater) {
            max_exp = it->first;
            max_coeff = it->second;
        }
    }
    return {max_exp, max_coeff};
}

std::pair<Expression, Expression> polynomial_div(const Expression& offset, const Expression& stride) {
    if (symbolic::eq(offset, symbolic::zero())) {
        return {symbolic::zero(), symbolic::zero()};
    }

    // Collect symbols for polynomial conversion
    SymbolVec symbols;
    SymbolSet atom_set;
    for (auto& s : symbolic::atoms(offset)) atom_set.insert(s);
    for (auto& s : symbolic::atoms(stride)) atom_set.insert(s);
    for (auto& s : atom_set) symbols.push_back(s);

    auto poly_stride = polynomial(stride, symbols);
    if (poly_stride == SymEngine::null) {
        // Fallback to simple division if not a polynomial
        Expression div_expr = SymEngine::div(offset, stride);
        Expression expanded = symbolic::expand(div_expr);
        Expression quotient = symbolic::zero();
        auto process_term = [&](const Expression& term) {
            SymEngine::RCP<const SymEngine::Basic> num, den;
            SymEngine::as_numer_denom(term, SymEngine::outArg(num), SymEngine::outArg(den));
            if (symbolic::eq(den, symbolic::one())) {
                quotient = symbolic::add(quotient, term);
            }
        };
        if (SymEngine::is_a<SymEngine::Add>(*expanded)) {
            auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expanded);
            for (auto& term : add->get_args()) process_term(term);
        } else {
            process_term(expanded);
        }
        Expression remainder = symbolic::sub(offset, symbolic::mul(quotient, stride));
        remainder = symbolic::expand(remainder);
        return {quotient, remainder};
    }

    Expression quotient_expr = symbolic::zero();
    Expression remainder_expr = symbolic::zero();
    Expression dividend_expr = offset;

    int max_iter = 100;
    size_t prev_term_count = std::numeric_limits<size_t>::max();
    while (!symbolic::eq(dividend_expr, symbolic::zero()) && max_iter-- > 0) {
        auto poly_dividend = polynomial(dividend_expr, symbols);
        if (poly_dividend == SymEngine::null) {
            break;
        }

        size_t cur_term_count = poly_dividend->get_poly().dict_.size();
        if (cur_term_count >= prev_term_count) {
            break;
        }
        prev_term_count = cur_term_count;

        auto [exp_div, coeff_div] = get_leading_term(poly_dividend);
        auto [exp_sor, coeff_sor] = get_leading_term(poly_stride);

        if (exp_div.empty() && symbolic::eq(coeff_div, symbolic::zero())) {
            break;
        }

        bool divisible = true;
        std::vector<int> exp_diff(exp_div.size());
        for (size_t i = 0; i < exp_div.size(); ++i) {
            if (exp_div[i] < exp_sor[i]) {
                divisible = false;
                break;
            }
            exp_diff[i] = exp_div[i] - exp_sor[i];
        }

        Expression term = symbolic::zero();
        if (divisible) {
            Expression coeff_q = symbolic::div(coeff_div, coeff_sor);
            if (symbolic::eq(coeff_q, symbolic::zero())) {
                divisible = false;
            } else {
                term = coeff_q;
                for (size_t i = 0; i < exp_diff.size(); ++i) {
                    if (exp_diff[i] > 0) {
                        term = symbolic::mul(term, symbolic::pow(symbols[i], symbolic::integer(exp_diff[i])));
                    }
                }
            }
        }

        if (divisible) {
            quotient_expr = symbolic::add(quotient_expr, term);
            dividend_expr = symbolic::sub(dividend_expr, symbolic::mul(term, stride));
            dividend_expr = symbolic::expand(dividend_expr);
        } else {
            // Move LT to remainder
            term = coeff_div;
            for (size_t i = 0; i < exp_div.size(); ++i) {
                if (exp_div[i] > 0) {
                    term = symbolic::mul(term, symbolic::pow(symbols[i], symbolic::integer(exp_div[i])));
                }
            }
            remainder_expr = symbolic::add(remainder_expr, term);
            dividend_expr = symbolic::sub(dividend_expr, term);
            dividend_expr = symbolic::expand(dividend_expr);
        }
    }
    remainder_expr = symbolic::add(remainder_expr, dividend_expr);
    remainder_expr = symbolic::expand(remainder_expr);

    return {quotient_expr, remainder_expr};
}

} // namespace symbolic
} // namespace sdfg
