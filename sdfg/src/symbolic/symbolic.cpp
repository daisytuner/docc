#include "sdfg/symbolic/symbolic.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <string>

#include <symengine/parser.h>
#include <symengine/subs.h>
#include "sdfg/exceptions.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "symengine/functions.h"
#include "symengine/logic.h"


namespace sdfg {
namespace symbolic {

Symbol symbol(const std::string& name) {
    if (name == "null") {
        throw InvalidSDFGException("null is not a valid symbol");
    } else if (name == "NULL") {
        throw InvalidSDFGException("NULL is not a valid symbol");
    } else if (name == "nullptr") {
        throw InvalidSDFGException("nullptr is not a valid symbol");
    }

    return SymEngine::symbol(name);
};

Integer integer(int64_t value) { return SymEngine::integer(value); };

Integer zero() { return symbolic::integer(0); };

Integer one() { return symbolic::integer(1); };

Condition __false__() { return SymEngine::boolean(false); };

Condition __true__() { return SymEngine::boolean(true); };

Symbol __nullptr__() { return SymEngine::symbol("__daisy_nullptr"); };

bool is_nullptr(const Symbol symbol) { return symbol->get_name() == "__daisy_nullptr"; };

bool is_pointer(const Symbol symbol) { return is_nullptr(symbol); };

bool is_nv(const Symbol symbol) {
    if (symbol == threadIdx_x() || symbol == threadIdx_y() || symbol == threadIdx_z() || symbol == blockIdx_x() ||
        symbol == blockIdx_y() || symbol == blockIdx_z() || symbol == blockDim_x() || symbol == blockDim_y() ||
        symbol == blockDim_z() || symbol == gridDim_x() || symbol == gridDim_y() || symbol == gridDim_z()) {
        return true;
    } else {
        return false;
    }
};

Expression divide_ceil(const Expression dividend, const Expression divisor) {
    Expression result;
    SymEngine::set_basic params = SymEngine::free_symbols(*dividend);
    if (params.empty()) { // will simplify to a statically known result, ok to use ceiling
        result = SymEngine::ceiling(SymEngine::div(dividend, divisor));
    } else { // if we know it will get generated, do integer math to cause ceiling a runtime without using float
        result = symbolic::div(SymEngine::add(dividend, SymEngine::sub(divisor, one())), divisor);
    }
    return result;
}

/***** Logical Expressions *****/

Condition And(const Condition lhs, const Condition rhs) { return SymEngine::logical_and({lhs, rhs}); };

Condition Or(const Condition lhs, const Condition rhs) { return SymEngine::logical_or({lhs, rhs}); };

Condition Not(const Condition expr) { return expr->logical_not(); };

bool is_true(const Expression expr) { return SymEngine::eq(*SymEngine::boolTrue, *expr); };

bool is_false(const Expression expr) { return SymEngine::eq(*SymEngine::boolFalse, *expr); };

/***** Integer Functions *****/

Expression add(const Expression lhs, const Expression rhs) { return SymEngine::add(lhs, rhs); };

Expression sub(const Expression lhs, const Expression rhs) { return SymEngine::sub(lhs, rhs); };

Expression mul(const Expression lhs, const Expression rhs) { return SymEngine::mul(lhs, rhs); };

Expression div(const Expression lhs, const Expression rhs) {
    if (eq(rhs, integer(0))) {
        return SymEngine::function_symbol("idiv", {lhs, rhs});
    }

    if (eq(rhs, integer(1))) {
        return lhs;
    }
    if (eq(lhs, integer(0))) {
        return integer(0);
    }
    if (SymEngine::is_a<SymEngine::Integer>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
        try {
            auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
            auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
            return integer(a / b);
        } catch (const SymEngine::SymEngineException&) {
            // Integer too large for long long, keep symbolic representation
            return SymEngine::function_symbol("idiv", {lhs, rhs});
        }
    }

    return SymEngine::function_symbol("idiv", {lhs, rhs});
};

Expression min(const Expression lhs, const Expression rhs) { return SymEngine::min({lhs, rhs}); };

Expression max(const Expression lhs, const Expression rhs) { return SymEngine::max({lhs, rhs}); };

Expression abs(const Expression expr) {
    auto abs = SymEngine::function_symbol("iabs", {expr});
    return abs;
};

Expression mod(const Expression lhs, const Expression rhs) {
    auto mod = SymEngine::function_symbol("imod", {lhs, rhs});
    return mod;
};

Expression pow(const Expression base, const Expression exp) { return SymEngine::pow(base, exp); };

Expression zext_i64(const Expression expr) {
    auto zext = SymEngine::make_rcp<ZExtI64Function>(expr);
    return zext;
}

Expression trunc_i32(const Expression expr) {
    auto trunc = SymEngine::make_rcp<TruncI32Function>(expr);
    return trunc;
}

Expression size_of_type(const types::IType& type) {
    if (auto scalar = dynamic_cast<const types::Scalar*>(&type)) {
        return integer((types::bit_width(scalar->primitive_type()) + 7) / 8);
    }
    auto so = SymEngine::make_rcp<SizeOfTypeFunction>(type);
    return so;
}

Expression dynamic_sizeof(const Symbol symbol) {
    auto so = SymEngine::make_rcp<DynamicSizeOfFunction>(symbol);
    return so;
}

Expression malloc_usable_size(const Symbol symbol) {
    auto mus = SymEngine::make_rcp<MallocUsableSizeFunction>(symbol);
    return mus;
}

/***** Comparisions *****/

Condition Eq(const Expression lhs, const Expression rhs) { return SymEngine::Eq(lhs, rhs); };

Condition Ne(const Expression lhs, const Expression rhs) { return SymEngine::Ne(lhs, rhs); };

Condition Lt(const Expression lhs, const Expression rhs) { return SymEngine::Lt(lhs, rhs); };

Condition Gt(const Expression lhs, const Expression rhs) { return SymEngine::Gt(lhs, rhs); };

Condition Le(const Expression lhs, const Expression rhs) { return SymEngine::Le(lhs, rhs); };

Condition Ge(const Expression lhs, const Expression rhs) { return SymEngine::Ge(lhs, rhs); };

/***** Modification *****/

Expression expand(const Expression expr) {
    auto new_expr = SymEngine::expand(expr);
    return new_expr;
};

Expression factor(const Expression expr) {
    // Atomic forms — nothing to factor
    if (SymEngine::is_a<SymEngine::Integer>(*expr) || SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        return expr;
    }

    // Factor inside Pow: factor the base
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto args = expr->get_args();
        if (args.size() == 2) {
            auto factored_base = factor(args[0]);
            if (!eq(factored_base, args[0])) {
                return symbolic::pow(factored_base, args[1]);
            }
        }
        return expr;
    }

    // Factor inside Mul: factor each argument
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto args = expr->get_args();
        Expression result = symbolic::one();
        bool changed = false;
        for (const auto& arg : args) {
            auto factored = factor(arg);
            if (!eq(factored, arg)) changed = true;
            result = symbolic::mul(result, factored);
        }
        return changed ? result : expr;
    }

    // Main case: Add (polynomial)
    if (!SymEngine::is_a<SymEngine::Add>(*expr)) {
        return expr;
    }

    auto symbol_set = atoms(expr);
    if (symbol_set.empty()) return expr;

    // Try each symbol as the polynomial variable
    for (const auto& sym : symbol_set) {
        SymbolVec gens = {sym};
        auto poly = polynomial(expr, gens);
        if (poly.is_null()) continue;

        auto& dict = poly->get_poly().get_dict();

        // Collect degree and coefficients
        int degree = 0;
        for (const auto& [exp, coeff] : dict) {
            degree = std::max(degree, static_cast<int>(exp[0]));
        }
        if (degree < 2 || degree > 20) continue;

        std::vector<Expression> sym_coeffs(degree + 1, symbolic::zero());
        for (const auto& [exp, coeff] : dict) {
            sym_coeffs[exp[0]] = coeff;
        }

        // Check if all coefficients are integers
        bool all_integer = true;
        std::vector<int64_t> int_coeffs(degree + 1, 0);
        for (int i = 0; i <= degree; i++) {
            if (SymEngine::is_a<SymEngine::Integer>(*sym_coeffs[i])) {
                try {
                    int_coeffs[i] = SymEngine::rcp_static_cast<const SymEngine::Integer>(sym_coeffs[i])->as_int();
                } catch (...) {
                    all_integer = false;
                    break;
                }
            } else if (!eq(sym_coeffs[i], symbolic::zero())) {
                all_integer = false;
            }
        }

        Expression common_factor = symbolic::one();

        // If not all integer, try extracting a common symbolic factor
        if (!all_integer) {
            Expression gcd_candidate = SymEngine::null;
            size_t min_complexity = SIZE_MAX;
            for (int i = 0; i <= degree; i++) {
                if (eq(sym_coeffs[i], symbolic::zero())) continue;
                if (SymEngine::is_a<SymEngine::Integer>(*sym_coeffs[i])) continue;
                size_t complexity = atoms(sym_coeffs[i]).size();
                if (complexity < min_complexity) {
                    min_complexity = complexity;
                    gcd_candidate = sym_coeffs[i];
                }
            }

            if (!gcd_candidate.is_null()) {
                bool all_divide = true;
                std::vector<Expression> reduced(degree + 1, symbolic::zero());
                for (int i = 0; i <= degree; i++) {
                    if (eq(sym_coeffs[i], symbolic::zero())) continue;
                    auto [q, r] = polynomial_div(sym_coeffs[i], gcd_candidate);
                    if (!eq(r, symbolic::zero())) {
                        all_divide = false;
                        break;
                    }
                    reduced[i] = q;
                }

                if (all_divide) {
                    all_integer = true;
                    for (int i = 0; i <= degree; i++) {
                        if (SymEngine::is_a<SymEngine::Integer>(*reduced[i])) {
                            try {
                                int_coeffs[i] =
                                    SymEngine::rcp_static_cast<const SymEngine::Integer>(reduced[i])->as_int();
                            } catch (...) {
                                all_integer = false;
                                break;
                            }
                        } else if (!eq(reduced[i], symbolic::zero())) {
                            all_integer = false;
                        }
                    }
                    if (all_integer) {
                        common_factor = gcd_candidate;
                    }
                }
            }

            if (!all_integer) continue;
        }

        // Extract GCD of integer coefficients
        int64_t coeff_gcd = 0;
        for (int i = 0; i <= degree; i++) {
            coeff_gcd = std::gcd(coeff_gcd, std::abs(int_coeffs[i]));
        }
        if (coeff_gcd > 1) {
            for (int i = 0; i <= degree; i++) {
                int_coeffs[i] /= coeff_gcd;
            }
            common_factor = symbolic::mul(common_factor, symbolic::integer(coeff_gcd));
        }

        // Ensure leading coefficient is positive
        if (int_coeffs[degree] < 0) {
            for (int i = 0; i <= degree; i++) {
                int_coeffs[i] = -int_coeffs[i];
            }
            common_factor = symbolic::mul(common_factor, symbolic::integer(-1));
        }

        // Factor integer-coefficient polynomial via integer root finding
        int current_degree = degree;
        std::vector<int64_t> current(int_coeffs.begin(), int_coeffs.end());
        Expression factored_result = common_factor;
        bool made_progress = false;

        while (current_degree >= 1) {
            if (current[0] == 0) {
                // x = 0 is a root
                int mult = 0;
                while (current_degree >= 1 && current[0] == 0) {
                    mult++;
                    for (int i = 0; i < current_degree; i++) {
                        current[i] = current[i + 1];
                    }
                    current_degree--;
                    current.resize(current_degree + 1);
                }
                factored_result = symbolic::mul(factored_result, symbolic::pow(sym, symbolic::integer(mult)));
                made_progress = true;
                continue;
            }

            // Candidate roots: divisors of |constant term|
            int64_t abs_a0 = std::abs(current[0]);
            std::vector<int64_t> divisors;
            for (int64_t d = 1; d * d <= abs_a0 && divisors.size() < 200; d++) {
                if (abs_a0 % d == 0) {
                    divisors.push_back(d);
                    if (d != abs_a0 / d) {
                        divisors.push_back(abs_a0 / d);
                    }
                }
            }

            bool found_root = false;
            for (int64_t d : divisors) {
                for (int64_t root : {d, -d}) {
                    // Horner evaluation
                    int64_t val = current[current_degree];
                    bool overflow = false;
                    for (int i = current_degree - 1; i >= 0; i--) {
                        if (root != 0 && std::abs(val) > (int64_t) 1e15 / (std::abs(root) + 1)) {
                            overflow = true;
                            break;
                        }
                        val = val * root + current[i];
                    }
                    if (overflow || val != 0) continue;

                    // Found root — extract with multiplicity via synthetic division
                    int mult = 0;
                    while (current_degree >= 1) {
                        std::vector<int64_t> quotient(current_degree, 0);
                        quotient[current_degree - 1] = current[current_degree];
                        for (int i = current_degree - 1; i >= 1; i--) {
                            quotient[i - 1] = current[i] + root * quotient[i];
                        }
                        int64_t rem = current[0] + root * quotient[0];
                        if (rem != 0) break;

                        mult++;
                        current_degree--;
                        current = quotient;
                    }

                    factored_result = symbolic::
                        mul(factored_result,
                            symbolic::pow(symbolic::sub(sym, symbolic::integer(root)), symbolic::integer(mult)));
                    made_progress = true;
                    found_root = true;
                    break;
                }
                if (found_root) break;
            }

            if (!found_root) break;
        }

        // Check remaining degree-2 for perfect square: a*x^2 + b*x + c with b^2 = 4ac
        if (current_degree == 2) {
            int64_t a = current[2], b = current[1], c = current[0];
            if (std::abs(b) < (int64_t) 1e9 && std::abs(a) < (int64_t) 1e9 && std::abs(c) < (int64_t) 1e9) {
                int64_t disc = b * b - 4 * a * c;
                if (disc == 0 && a > 0) {
                    int64_t d = static_cast<int64_t>(std::round(std::sqrt(static_cast<double>(a))));
                    if (d > 0 && d * d == a && b % (2 * d) == 0) {
                        int64_t e = b / (2 * d);
                        if (e * e == c) {
                            Expression base =
                                (d == 1)
                                    ? symbolic::add(sym, symbolic::integer(e))
                                    : symbolic::add(symbolic::mul(symbolic::integer(d), sym), symbolic::integer(e));
                            factored_result = symbolic::mul(factored_result, symbolic::pow(base, symbolic::integer(2)));
                            current_degree = 0;
                            made_progress = true;
                        }
                    }
                }
            }
        }

        if (!made_progress) continue;

        // Add remaining unfactored polynomial
        if (current_degree >= 1) {
            Expression remaining = symbolic::zero();
            for (int i = current_degree; i >= 0; i--) {
                if (current[i] == 0) continue;
                if (i == 0) {
                    remaining = symbolic::add(remaining, symbolic::integer(current[i]));
                } else if (i == 1) {
                    remaining = symbolic::add(remaining, symbolic::mul(symbolic::integer(current[i]), sym));
                } else {
                    remaining = symbolic::
                        add(remaining,
                            symbolic::mul(symbolic::integer(current[i]), symbolic::pow(sym, symbolic::integer(i))));
                }
            }
            factored_result = symbolic::mul(factored_result, remaining);
        } else if (current_degree == 0 && current[0] != 1) {
            factored_result = symbolic::mul(factored_result, symbolic::integer(current[0]));
        }

        return factored_result;
    }

    return expr;
}

namespace {

/// Decompose expr into (base, integer_offset) where expr = base + offset.
/// If expr is purely an integer, base = integer(0).
/// If expr has no integer addend, offset = 0.
std::pair<Expression, int64_t> decompose_offset(const Expression& expr) {
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        try {
            return {symbolic::zero(), SymEngine::rcp_static_cast<const SymEngine::Integer>(expr)->as_int()};
        } catch (const SymEngine::SymEngineException&) {
            return {expr, 0};
        }
    }
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        auto constant = add->get_coef();
        if (SymEngine::is_a<SymEngine::Integer>(*constant)) {
            try {
                int64_t offset = SymEngine::rcp_static_cast<const SymEngine::Integer>(constant)->as_int();
                if (offset != 0) {
                    auto base = SymEngine::sub(expr, constant);
                    return {base, offset};
                }
            } catch (const SymEngine::SymEngineException&) {
                // too large
            }
        }
    }
    return {expr, 0};
}

/// Simplify min/max by grouping args that share the same base expression
/// and keeping only the extreme offset per group.
/// For max: keep the largest offset per base → max(i, i-1, i+1) = i+1
/// For min: keep the smallest offset per base → min(i, i-1, i+1) = i-1
template<bool IsMax>
Expression simplify_minmax(const SymEngine::vec_basic& args) {
    // Map from base expression to best (offset, original_expr)
    std::vector<std::pair<Expression, std::pair<int64_t, Expression>>> groups;

    for (const auto& arg : args) {
        auto [base, offset] = decompose_offset(arg);

        bool found = false;
        for (auto& [g_base, g_best] : groups) {
            if (symbolic::eq(g_base, base)) {
                if constexpr (IsMax) {
                    if (offset > g_best.first) {
                        g_best = {offset, arg};
                    }
                } else {
                    if (offset < g_best.first) {
                        g_best = {offset, arg};
                    }
                }
                found = true;
                break;
            }
        }
        if (!found) {
            groups.push_back({base, {offset, arg}});
        }
    }

    // If we eliminated any args, rebuild
    if (groups.size() < args.size()) {
        if (groups.size() == 1) {
            return groups[0].second.second;
        }
        SymEngine::vec_basic new_args;
        for (auto& [_, best] : groups) {
            new_args.push_back(best.second);
        }
        if constexpr (IsMax) {
            return SymEngine::max(new_args);
        } else {
            return SymEngine::min(new_args);
        }
    }

    return SymEngine::null;
}

} // anonymous namespace

Expression simplify(const Expression expr) {
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*expr)) {
        auto slt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(expr);
        auto lhs = slt->get_arg1();
        auto rhs = slt->get_arg2();
        auto simple_lhs = symbolic::simplify(lhs);
        auto simple_rhs = symbolic::simplify(rhs);
        return symbolic::Lt(simple_lhs, simple_rhs);
    }
    if (SymEngine::is_a<SymEngine::LessThan>(*expr)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(expr);
        auto lhs = le->get_arg1();
        auto rhs = le->get_arg2();
        auto simple_lhs = symbolic::simplify(lhs);
        auto simple_rhs = symbolic::simplify(rhs);
        return symbolic::Le(simple_lhs, simple_rhs);
    }

    // Simplify max(a, a+c1, a+c2, ...) by keeping only the largest offset per base
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(expr);
        auto args = max_op->get_args();
        // First simplify each arg
        SymEngine::vec_basic simple_args;
        for (const auto& arg : args) {
            simple_args.push_back(symbolic::simplify(arg));
        }
        auto result = simplify_minmax<true>(simple_args);
        if (!result.is_null()) {
            return result;
        }
        return SymEngine::max(simple_args);
    }

    // Simplify min(a, a+c1, a+c2, ...) by keeping only the smallest offset per base
    if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto min_op = SymEngine::rcp_static_cast<const SymEngine::Min>(expr);
        auto args = min_op->get_args();
        // First simplify each arg
        SymEngine::vec_basic simple_args;
        for (const auto& arg : args) {
            simple_args.push_back(symbolic::simplify(arg));
        }
        auto result = simplify_minmax<false>(simple_args);
        if (!result.is_null()) {
            return result;
        }
        return SymEngine::min(simple_args);
    }

    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        auto args = add->get_args();
        for (const auto& arg : args) {
            if (SymEngine::is_a<SymEngine::Max>(*arg)) {
                auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(arg);
                auto max_args = max_op->get_args();

                std::vector<Expression> other_args;
                bool skipped = false;
                for (const auto& a : args) {
                    if (eq(a, arg) && !skipped) {
                        skipped = true;
                    } else {
                        other_args.push_back(a);
                    }
                }
                auto rest = SymEngine::add(other_args);

                SymEngine::vec_basic new_max_args;
                for (const auto& m_arg : max_args) {
                    new_max_args.push_back(symbolic::simplify(SymEngine::add(rest, m_arg)));
                }
                return SymEngine::max(new_max_args);
            }
        }
    }

    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        auto func_sym = SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(expr);
        auto func_id = func_sym->get_name();
        if (func_id == "idiv") {
            auto lhs = func_sym->get_args()[0];
            auto rhs = func_sym->get_args()[1];
            if (symbolic::eq(rhs, symbolic::integer(0))) {
                return expr;
            }
            if (symbolic::is_true(symbolic::Lt(lhs, rhs))) {
                return symbolic::zero();
            }

            if (SymEngine::is_a<SymEngine::Mul>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
                auto lhs_mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(lhs);
                auto rhs_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs);
                auto lhs_args = lhs_mul->get_args();

                bool skipped = false;
                Expression new_mul = SymEngine::integer(1);
                for (auto& arg : lhs_args) {
                    if (eq(arg, rhs_int) && !skipped) {
                        skipped = true;
                    } else {
                        new_mul = SymEngine::mul(new_mul, arg);
                    }
                }
                if (skipped) {
                    return new_mul;
                }
            } else if (SymEngine::is_a<SymEngine::Integer>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
                try {
                    auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
                    auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
                    return integer(a / b);
                } catch (const SymEngine::SymEngineException&) {
                    // Integer too large, cannot simplify - fall through
                }
            }
        } else if (func_id == "imod") {
            auto lhs = func_sym->get_args()[0];
            auto rhs = func_sym->get_args()[1];
            if (SymEngine::is_a<SymEngine::Integer>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
                try {
                    auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
                    auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
                    return integer(a % b);
                } catch (const SymEngine::SymEngineException&) {
                    // Integer too large, cannot simplify - fall through
                }
            }
        } else if (func_id == "zext_i64") {
            auto arg = func_sym->get_args()[0];
            auto simple_arg = symbolic::simplify(arg);

            bool non_negative = false;
            if (SymEngine::is_a<SymEngine::Integer>(*simple_arg)) {
                try {
                    if (SymEngine::rcp_static_cast<const SymEngine::Integer>(simple_arg)->as_int() >= 0) {
                        non_negative = true;
                    }
                } catch (const SymEngine::SymEngineException&) {
                    // Integer too large to check, assume not simplifiable
                }
            } else if (SymEngine::is_a<SymEngine::Max>(*simple_arg)) {
                auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(simple_arg);
                for (const auto& m_arg : max_op->get_args()) {
                    if (SymEngine::is_a<SymEngine::Integer>(*m_arg)) {
                        try {
                            if (SymEngine::rcp_static_cast<const SymEngine::Integer>(m_arg)->as_int() >= 0) {
                                non_negative = true;
                                break;
                            }
                        } catch (const SymEngine::SymEngineException&) {
                            // Integer too large, skip this arg
                        }
                    }
                }
            }

            if (non_negative) {
                return simple_arg;
            }

            if (!eq(arg, simple_arg)) {
                return zext_i64(simple_arg);
            }
        } else if (func_id == "iabs") {
            auto arg = func_sym->get_args()[0];
            auto simple_arg = symbolic::simplify(arg);
            if (SymEngine::is_a<SymEngine::Integer>(*simple_arg)) {
                try {
                    auto val = SymEngine::rcp_static_cast<const SymEngine::Integer>(simple_arg)->as_int();
                    return integer(val >= 0 ? val : -val);
                } catch (const SymEngine::SymEngineException&) {
                    // Integer too large, cannot simplify - fall through
                }
            }
        }
    }

    try {
        auto new_expr = SymEngine::simplify(expr);
        return new_expr;
    } catch (const SymEngine::SymEngineException& e) {
        return expr;
    }
};

Expression overapproximate(const Expression expr) {
    // For min: pick the arg with the largest integer offset (upper bound of min)
    if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto min_op = SymEngine::rcp_static_cast<const SymEngine::Min>(expr);
        auto args = min_op->get_args();

        // First overapproximate each arg recursively
        SymEngine::vec_basic approx_args;
        for (const auto& arg : args) {
            approx_args.push_back(overapproximate(arg));
        }

        // Find the arg with purely integer value (prefer constant over symbolic)
        Expression best = SymEngine::null;
        int64_t best_val = std::numeric_limits<int64_t>::min();
        bool has_integer = false;
        for (const auto& arg : approx_args) {
            if (SymEngine::is_a<SymEngine::Integer>(*arg)) {
                try {
                    auto val = SymEngine::rcp_static_cast<const SymEngine::Integer>(arg)->as_int();
                    if (!has_integer || val > best_val) {
                        best = arg;
                        best_val = val;
                        has_integer = true;
                    }
                } catch (const SymEngine::SymEngineException&) {
                }
            }
        }
        if (has_integer) {
            return best;
        }

        // No pure integer arg — group by base, pick largest offset per group
        auto result = simplify_minmax<true>(approx_args);
        if (!result.is_null()) {
            return result;
        }
        return SymEngine::min(approx_args);
    }

    // For max: pick the arg with the largest integer offset (upper bound of max)
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(expr);
        auto args = max_op->get_args();

        SymEngine::vec_basic approx_args;
        for (const auto& arg : args) {
            approx_args.push_back(overapproximate(arg));
        }

        Expression best = SymEngine::null;
        int64_t best_val = std::numeric_limits<int64_t>::min();
        bool has_integer = false;
        for (const auto& arg : approx_args) {
            if (SymEngine::is_a<SymEngine::Integer>(*arg)) {
                try {
                    auto val = SymEngine::rcp_static_cast<const SymEngine::Integer>(arg)->as_int();
                    if (!has_integer || val > best_val) {
                        best = arg;
                        best_val = val;
                        has_integer = true;
                    }
                } catch (const SymEngine::SymEngineException&) {
                }
            }
        }
        if (has_integer) {
            return best;
        }

        auto result = simplify_minmax<true>(approx_args);
        if (!result.is_null()) {
            return result;
        }
        return SymEngine::max(approx_args);
    }

    // Recurse into Add
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add_node = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        auto args = add_node->get_args();

        // If exactly one addend is a Min, distribute the rest into it:
        // min(a, b) + c = min(a+c, b+c)
        // This exposes pure integer args after cancellation.
        SymEngine::vec_basic min_children;
        SymEngine::vec_basic rest;
        int min_count = 0;
        for (const auto& arg : args) {
            if (min_count == 0 && SymEngine::is_a<SymEngine::Min>(*arg)) {
                min_count++;
                auto min_op = SymEngine::rcp_static_cast<const SymEngine::Min>(arg);
                for (const auto& mc : min_op->get_args()) {
                    min_children.push_back(mc);
                }
            } else {
                rest.push_back(arg);
            }
        }

        if (min_count == 1 && !rest.empty()) {
            auto sum_rest = SymEngine::add(rest);
            SymEngine::vec_basic new_min_args;
            for (const auto& mc : min_children) {
                new_min_args.push_back(SymEngine::expand(SymEngine::add(mc, sum_rest)));
            }
            return overapproximate(SymEngine::min(new_min_args));
        }

        // Normal case: recurse into each addend
        SymEngine::vec_basic new_args;
        bool changed = false;
        for (const auto& arg : args) {
            auto new_arg = overapproximate(arg);
            new_args.push_back(new_arg);
            if (!symbolic::eq(arg, new_arg)) changed = true;
        }
        if (changed) {
            return SymEngine::add(new_args);
        }
    }

    // Recurse into Mul
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto mul_node = SymEngine::rcp_static_cast<const SymEngine::Mul>(expr);
        auto args = mul_node->get_args();
        SymEngine::vec_basic new_args;
        bool changed = false;
        for (const auto& arg : args) {
            auto new_arg = overapproximate(arg);
            new_args.push_back(new_arg);
            if (!symbolic::eq(arg, new_arg)) changed = true;
        }
        if (changed) {
            return SymEngine::mul(new_args);
        }
    }

    return expr;
};

bool eq(const Expression lhs, const Expression rhs) { return SymEngine::eq(*lhs, *rhs); };

bool null_safe_eq(const Expression lhs, const Expression rhs) {
    if (lhs.is_null() && rhs.is_null()) {
        return true;
    } else if (!lhs.is_null() && !rhs.is_null()) {
        return SymEngine::eq(*lhs, *rhs);
    } else {
        return false;
    }
}

bool uses(const Expression expr, const Symbol sym) { return SymEngine::has_symbol(*expr, *sym); };

bool uses(const Expression expr, const std::string& name) { return symbolic::uses(expr, symbol(name)); };

SymbolSet atoms(const Expression expr) {
    SymbolSet atoms;
    for (auto& atom : SymEngine::atoms<const SymEngine::Basic>(*expr)) {
        if (SymEngine::is_a<SymEngine::Symbol>(*atom)) {
            atoms.insert(SymEngine::rcp_static_cast<const SymEngine::Symbol>(atom));
        }
    }
    return atoms;
};

ExpressionSet muls(const Expression expr) { return SymEngine::atoms<const SymEngine::Mul>(*expr); };

Expression subs(const Expression expr, const Expression old_expr, const Expression new_expr) {
    SymEngine::map_basic_basic d;
    d[old_expr] = new_expr;

    return SymEngine::subs(expr, d);
};

Condition subs(const Condition expr, const Expression old_expr, const Expression new_expr) {
    SymEngine::map_basic_basic d;
    d[old_expr] = new_expr;

    return SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(subs(expr, d));
};

Expression parse(const std::string& expr_str) {
    auto expr = SymEngine::parse(expr_str);
    expr = symbolic::subs(expr, symbolic::symbol("true"), symbolic::one());
    expr = symbolic::subs(expr, symbolic::symbol("false"), symbolic::zero());
    return expr;
};

Expression inverse(const Expression expr, const Symbol symbol) {
    // Currently only affine inverse is supported
    SymbolVec symbols = {symbol};
    Polynomial poly = polynomial(expr, symbols);
    if (poly.is_null()) {
        return SymEngine::null;
    }
    AffineCoeffs affine_coeffs = affine_coefficients(poly);
    return affine_inverse(affine_coeffs, symbol);
}

/***** NV Symbols *****/

Symbol threadIdx_x() { return symbol("threadIdx.x"); };

Symbol threadIdx_y() { return symbol("threadIdx.y"); };

Symbol threadIdx_z() { return symbol("threadIdx.z"); };

Symbol blockDim_x() { return symbol("blockDim.x"); };

Symbol blockDim_y() { return symbol("blockDim.y"); };

Symbol blockDim_z() { return symbol("blockDim.z"); };

Symbol blockIdx_x() { return symbol("blockIdx.x"); };

Symbol blockIdx_y() { return symbol("blockIdx.y"); };

Symbol blockIdx_z() { return symbol("blockIdx.z"); };

Symbol gridDim_x() { return symbol("gridDim.x"); };

Symbol gridDim_y() { return symbol("gridDim.y"); };

Symbol gridDim_z() { return symbol("gridDim.z"); }

bool has_dynamic_sizeof(const Expression expr) {
    for (auto& func : SymEngine::atoms<SymEngine::FunctionSymbol>(*expr)) {
        if (SymEngine::is_a<DynamicSizeOfFunction>(*func) &&
            SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(func)->get_name() == "dynamic_sizeof") {
            return true;
        }
    }
    return false;
};

} // namespace symbolic
} // namespace sdfg
