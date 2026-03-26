#include "sdfg/symbolic/extreme_values.h"

#include "sdfg/symbolic/symbolic.h"
#include "symengine/basic.h"
#include "symengine/functions.h"

namespace sdfg {
namespace symbolic {

size_t MAX_DEPTH = 100;
Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth);
Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth);

Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth) {
    // Base Cases
    if (depth > MAX_DEPTH) {
        return SymEngine::null;
    }

    if (SymEngine::is_a<SymEngine::NaN>(*expr)) {
        return SymEngine::null;
    }

    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::Infty>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<symbolic::ZExtI64Function>(*expr)) {
        auto zext = SymEngine::rcp_static_cast<const symbolic::ZExtI64Function>(expr);
        auto min_arg = minimum(zext->get_args()[0], parameters, assumptions, depth + 1);
        if (min_arg == SymEngine::null) {
            return SymEngine::null;
        } else {
            return symbolic::zext_i64(min_arg);
        }
    } else if (SymEngine::is_a<symbolic::TruncI32Function>(*expr)) {
        auto trunc = SymEngine::rcp_static_cast<const symbolic::TruncI32Function>(expr);
        auto min_arg = minimum(trunc->get_args()[0], parameters, assumptions, depth + 1);
        if (min_arg == SymEngine::null) {
            return SymEngine::null;
        } else {
            return symbolic::trunc_i32(min_arg);
        }
    }

    // Symbol
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (parameters.find(sym) != parameters.end()) {
            return sym;
        }
        if (assumptions.find(sym) != assumptions.end()) {
            return minimum(assumptions.at(sym).lower_bound_deprecated(), parameters, assumptions, depth + 1);
        }
        return SymEngine::null;
    }

    // Pow(base, k) with constant integer exponent k
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow_expr = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto args = pow_expr->get_args();
        if (args.size() != 2 || !SymEngine::is_a<SymEngine::Integer>(*args[1])) {
            return SymEngine::null;
        }

        long long exp_val = 0;
        try {
            exp_val = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
        } catch (const SymEngine::SymEngineException&) {
            return SymEngine::null;
        }

        if (exp_val < 0) {
            return SymEngine::null;
        }
        if (exp_val == 0) {
            return symbolic::one();
        }

        auto base_lb = minimum(args[0], parameters, assumptions, depth + 1);
        auto base_ub = maximum(args[0], parameters, assumptions, depth + 1);
        if (base_lb == SymEngine::null || base_ub == SymEngine::null) {
            return SymEngine::null;
        }

        auto exp_expr = symbolic::integer(exp_val);
        auto lb_pow = symbolic::pow(base_lb, exp_expr);
        auto ub_pow = symbolic::pow(base_ub, exp_expr);

        // Odd powers are monotonic. Even powers need interval sign reasoning.
        if (exp_val % 2 != 0) {
            return lb_pow;
        }

        auto zero = symbolic::zero();
        bool interval_nonneg = symbolic::is_true(symbolic::Ge(base_lb, zero));
        bool interval_nonpos = symbolic::is_true(symbolic::Le(base_ub, zero));
        bool crosses_zero = symbolic::is_true(symbolic::Le(base_lb, zero)) &&
                            symbolic::is_true(symbolic::Ge(base_ub, zero));

        if (crosses_zero) {
            return zero;
        }
        if (interval_nonneg || interval_nonpos) {
            return symbolic::min(lb_pow, ub_pow);
        }

        // Cannot prove whether 0 belongs to the interval -> avoid unsound bound.
        return SymEngine::null;
    }

    // Mul
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(expr);
        const auto& args = mul->get_args();
        size_t n = args.size();

        std::vector<std::pair<Expression, Expression>> bounds;
        bounds.reserve(n);

        for (const auto& arg : args) {
            Expression min_val = minimum(arg, parameters, assumptions, depth + 1);
            Expression max_val = maximum(arg, parameters, assumptions, depth + 1);

            if (min_val == SymEngine::null || max_val == SymEngine::null) {
                return SymEngine::null;
            }
            bounds.emplace_back(min_val, max_val);
        }

        // Iterate over 2^n combinations
        Expression min_product = SymEngine::null;
        const size_t total_combinations = 1ULL << n;

        for (size_t mask = 0; mask < total_combinations; ++mask) {
            Expression product = SymEngine::integer(1);
            for (size_t i = 0; i < n; ++i) {
                const auto& bound = bounds[i];
                Expression val = (mask & (1ULL << i)) ? bound.second : bound.first;
                product = symbolic::mul(product, val);
            }
            if (min_product == SymEngine::null) {
                min_product = product;
            } else {
                min_product = symbolic::min(min_product, product);
            }
        }

        return min_product;
    }

    // Add
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        const auto& args = add->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum(arg, parameters, assumptions, depth + 1);
            if (lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (lbs == SymEngine::null) {
                lbs = lb;
            } else {
                lbs = symbolic::add(lbs, lb);
            }
        }
        return lbs;
    }

    // Max
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Max>(expr)->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum(arg, parameters, assumptions, depth + 1);
            if (lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (lbs == SymEngine::null) {
                lbs = lb;
            } else {
                lbs = symbolic::min(lbs, lb);
            }
        }
        return lbs;
    }

    // Min
    if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Min>(expr)->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum(arg, parameters, assumptions, depth + 1);
            if (lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (lbs == SymEngine::null) {
                lbs = lb;
            } else {
                lbs = symbolic::min(lbs, lb);
            }
        }
        return lbs;
    }

    return SymEngine::null;
}

Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth) {
    if (depth > MAX_DEPTH) {
        return SymEngine::null;
    }

    if (SymEngine::is_a<SymEngine::NaN>(*expr)) {
        return SymEngine::null;
    }

    // Base Cases
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return expr;
    } else if (SymEngine::is_a<SymEngine::Infty>(*expr)) {
        return expr;
    }

    // Symbol
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (parameters.find(sym) != parameters.end()) {
            return sym;
        }
        if (assumptions.find(sym) != assumptions.end()) {
            return maximum(assumptions.at(sym).upper_bound_deprecated(), parameters, assumptions, depth + 1);
        }
        return SymEngine::null;
    }

    // Pow(base, k) with constant integer exponent k
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow_expr = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto args = pow_expr->get_args();
        if (args.size() != 2 || !SymEngine::is_a<SymEngine::Integer>(*args[1])) {
            return SymEngine::null;
        }

        long long exp_val = 0;
        try {
            exp_val = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
        } catch (const SymEngine::SymEngineException&) {
            return SymEngine::null;
        }

        if (exp_val < 0) {
            return SymEngine::null;
        }
        if (exp_val == 0) {
            return symbolic::one();
        }

        auto base_lb = minimum(args[0], parameters, assumptions, depth + 1);
        auto base_ub = maximum(args[0], parameters, assumptions, depth + 1);
        if (base_lb == SymEngine::null || base_ub == SymEngine::null) {
            return SymEngine::null;
        }

        auto exp_expr = symbolic::integer(exp_val);
        auto lb_pow = symbolic::pow(base_lb, exp_expr);
        auto ub_pow = symbolic::pow(base_ub, exp_expr);

        if (exp_val % 2 != 0) {
            return ub_pow;
        }

        return symbolic::max(lb_pow, ub_pow);
    }

    if (SymEngine::is_a<symbolic::ZExtI64Function>(*expr)) {
        auto zext = SymEngine::rcp_static_cast<const symbolic::ZExtI64Function>(expr);
        auto max_arg = maximum(zext->get_args()[0], parameters, assumptions, depth + 1);
        if (max_arg == SymEngine::null) {
            return SymEngine::null;
        } else {
            return symbolic::zext_i64(max_arg);
        }
    }
    if (SymEngine::is_a<symbolic::TruncI32Function>(*expr)) {
        auto trunc = SymEngine::rcp_static_cast<const symbolic::TruncI32Function>(expr);
        auto max_arg = maximum(trunc->get_args()[0], parameters, assumptions, depth + 1);
        if (max_arg == SymEngine::null) {
            return SymEngine::null;
        } else {
            return symbolic::trunc_i32(max_arg);
        }
    }

    // Mul
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(expr);
        const auto& args = mul->get_args();
        size_t n = args.size();

        std::vector<std::pair<Expression, Expression>> bounds;
        bounds.reserve(n);

        for (const auto& arg : args) {
            Expression min_val = minimum(arg, parameters, assumptions, depth + 1);
            Expression max_val = maximum(arg, parameters, assumptions, depth + 1);

            if (min_val == SymEngine::null || max_val == SymEngine::null) {
                return SymEngine::null;
            }
            bounds.emplace_back(min_val, max_val);
        }

        // Iterate over 2^n combinations
        Expression max_product = SymEngine::null;
        const size_t total_combinations = 1ULL << n;

        for (size_t mask = 0; mask < total_combinations; ++mask) {
            Expression product = SymEngine::integer(1);
            for (size_t i = 0; i < n; ++i) {
                const auto& bound = bounds[i];
                Expression val = (mask & (1ULL << i)) ? bound.second : bound.first;
                product = symbolic::mul(product, val);
            }
            if (max_product == SymEngine::null) {
                max_product = product;
            } else {
                max_product = symbolic::max(max_product, product);
            }
        }

        return max_product;
    }

    // Add
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        const auto& args = add->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum(arg, parameters, assumptions, depth + 1);
            if (ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (ubs == SymEngine::null) {
                ubs = ub;
            } else {
                ubs = symbolic::add(ubs, ub);
            }
        }
        return ubs;
    }

    // Max
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Max>(expr)->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum(arg, parameters, assumptions, depth + 1);
            if (ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (ubs == SymEngine::null) {
                ubs = ub;
            } else {
                ubs = symbolic::max(ubs, ub);
            }
        }
        return ubs;
    }

    // Min
    if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Min>(expr)->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum(arg, parameters, assumptions, depth + 1);
            if (ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (ubs == SymEngine::null) {
                ubs = ub;
            } else {
                ubs = symbolic::max(ubs, ub);
            }
        }
        return ubs;
    }

    return SymEngine::null;
}

Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions) {
    return minimum(expr, parameters, assumptions, 0);
}

Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions) {
    return maximum(expr, parameters, assumptions, 0);
}

Expression minimum_new(
    const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth, bool tight
);
Expression maximum_new(
    const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth, bool tight
);

Expression minimum_new(
    const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth, bool tight
) {
    // End of recursion: fail
    if (depth > MAX_DEPTH) {
        return SymEngine::null;
    }
    if (SymEngine::is_a<SymEngine::NaN>(*expr)) {
        return SymEngine::null;
    }
    if (SymEngine::is_a<SymEngine::Infty>(*expr)) {
        return SymEngine::null;
    }
    // End of recursion: success
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return expr;
    }
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (parameters.find(sym) != parameters.end()) {
            return sym;
        }
    }

    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        auto func_sym = SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(expr);
        auto func_id = func_sym->get_name();
        if (func_id == "zext_i64") {
            auto zext = SymEngine::rcp_static_cast<const symbolic::ZExtI64Function>(expr);
            auto min_arg = minimum_new(zext->get_args()[0], parameters, assumptions, depth + 1, tight);
            if (min_arg == SymEngine::null) {
                return SymEngine::null;
            } else {
                return symbolic::zext_i64(min_arg);
            }
        } else if (func_id == "trunc_i32") {
            auto trunc = SymEngine::rcp_static_cast<const symbolic::TruncI32Function>(expr);
            auto min_arg = minimum_new(trunc->get_args()[0], parameters, assumptions, depth + 1, tight);
            if (min_arg == SymEngine::null) {
                return SymEngine::null;
            } else {
                return symbolic::trunc_i32(min_arg);
            }
        } else if (func_id == "idiv") {
            auto numerator = func_sym->get_args()[0];
            auto denominator = func_sym->get_args()[1];
            if (!SymEngine::is_a<const SymEngine::Integer>(*denominator)) {
                // Denominator is not a constant integer -> cannot soundly bound.
                return SymEngine::null;
            }

            auto numerator_lb = minimum_new(numerator, parameters, assumptions, depth + 1, tight);
            auto denominator_ub = maximum_new(denominator, parameters, assumptions, depth + 1, tight);
            if (numerator_lb == SymEngine::null || denominator_ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (symbolic::is_true(symbolic::Le(denominator_ub, symbolic::zero()))) {
                // Denominator can be zero or negative -> cannot soundly bound.
                return SymEngine::null;
            }
            return symbolic::div(numerator_lb, denominator_ub);
        } else if (func_id == "imod") {
            auto lhs = func_sym->get_args()[0];
            auto rhs = func_sym->get_args()[1];
            if (!SymEngine::is_a<const SymEngine::Integer>(*rhs)) {
                return SymEngine::null;
            }

            auto lhs_lb = minimum_new(lhs, parameters, assumptions, depth + 1, tight);
            auto lhs_ub = maximum_new(lhs, parameters, assumptions, depth + 1, tight);
            if (lhs_lb == SymEngine::null || lhs_ub == SymEngine::null) {
                return SymEngine::null;
            }

            // Handle negative cases: min can be -(|rhs| - 1)
            bool has_negative = symbolic::is_true(symbolic::Lt(lhs_lb, symbolic::zero())) ||
                                symbolic::is_true(symbolic::Lt(rhs, symbolic::zero()));
            auto neg_bound = symbolic::sub(symbolic::one(), symbolic::simplify(symbolic::abs(rhs)));
            symbolic::Expression zero = symbolic::zero();

            auto width = symbolic::sub(lhs_ub, lhs_lb);
            if (symbolic::is_true(symbolic::Lt(width, rhs))) {
                // Range doesn't span full modulus cycle
                bool wraps = symbolic::is_true(symbolic::Lt(symbolic::mod(lhs_ub, rhs), symbolic::mod(lhs_lb, rhs)));
                if (wraps) {
                    return has_negative ? neg_bound : zero;
                }
                return symbolic::simplify(symbolic::mod(lhs_lb, rhs));
            }

            // Range spans full cycle
            return has_negative ? neg_bound : zero;
        }
    }

    // Symbol
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (assumptions.find(sym) != assumptions.end()) {
            if (tight) {
                if (assumptions.at(sym).tight_lower_bound().is_null()) {
                    return SymEngine::null;
                }
                return minimum_new(assumptions.at(sym).tight_lower_bound(), parameters, assumptions, depth + 1, tight);
            }
            symbolic::Expression new_lb = SymEngine::null;
            for (auto& lb : assumptions.at(sym).lower_bounds()) {
                auto new_min = minimum_new(lb, parameters, assumptions, depth + 1, tight);
                if (new_min.is_null()) {
                    continue;
                }
                if (new_lb.is_null()) {
                    new_lb = new_min;
                    continue;
                }
                new_lb = symbolic::max(new_lb, new_min);
            }
            return new_lb;
        }
        return SymEngine::null;
    }

    // Pow(base, k) with constant integer exponent k
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow_expr = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto args = pow_expr->get_args();
        if (args.size() != 2 || !SymEngine::is_a<SymEngine::Integer>(*args[1])) {
            return SymEngine::null;
        }

        long long exp_val = 0;
        try {
            exp_val = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
        } catch (const SymEngine::SymEngineException&) {
            return SymEngine::null;
        }

        if (exp_val < 0) {
            return SymEngine::null;
        }
        if (exp_val == 0) {
            return symbolic::one();
        }

        auto base_lb = minimum_new(args[0], parameters, assumptions, depth + 1, tight);
        auto base_ub = maximum_new(args[0], parameters, assumptions, depth + 1, tight);
        if (base_lb == SymEngine::null || base_ub == SymEngine::null) {
            return SymEngine::null;
        }

        auto exp_expr = symbolic::integer(exp_val);
        auto lb_pow = symbolic::pow(base_lb, exp_expr);
        auto ub_pow = symbolic::pow(base_ub, exp_expr);

        // Odd powers are monotonic. Even powers need interval sign reasoning.
        if (exp_val % 2 != 0) {
            return lb_pow;
        }

        auto zero = symbolic::zero();
        bool interval_nonneg = symbolic::is_true(symbolic::Ge(base_lb, zero));
        bool interval_nonpos = symbolic::is_true(symbolic::Le(base_ub, zero));
        bool crosses_zero = symbolic::is_true(symbolic::Le(base_lb, zero)) &&
                            symbolic::is_true(symbolic::Ge(base_ub, zero));

        if (crosses_zero) {
            return zero;
        }
        if (interval_nonneg || interval_nonpos) {
            return symbolic::min(lb_pow, ub_pow);
        }

        // Cannot prove whether 0 belongs to the interval -> avoid unsound bound.
        return SymEngine::null;
    }

    // Mul
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(expr);
        const auto& args = mul->get_args();
        size_t n = args.size();

        std::vector<std::pair<Expression, Expression>> bounds;
        bounds.reserve(n);

        for (const auto& arg : args) {
            Expression min_val = minimum_new(arg, parameters, assumptions, depth + 1, tight);
            Expression max_val = maximum_new(arg, parameters, assumptions, depth + 1, tight);

            if (min_val == SymEngine::null || max_val == SymEngine::null) {
                return SymEngine::null;
            }
            bounds.emplace_back(min_val, max_val);
        }

        // Iterate over 2^n combinations
        Expression min_product = SymEngine::null;
        const size_t total_combinations = 1ULL << n;

        for (size_t mask = 0; mask < total_combinations; ++mask) {
            Expression product = SymEngine::integer(1);
            for (size_t i = 0; i < n; ++i) {
                const auto& bound = bounds[i];
                Expression val = (mask & (1ULL << i)) ? bound.second : bound.first;
                product = symbolic::mul(product, val);
            }
            if (min_product == SymEngine::null) {
                min_product = product;
            } else {
                min_product = symbolic::min(min_product, product);
            }
        }

        return min_product;
    }

    // Add
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        const auto& args = add->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum_new(arg, parameters, assumptions, depth + 1, tight);
            if (lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (lbs == SymEngine::null) {
                lbs = lb;
            } else {
                lbs = symbolic::add(lbs, lb);
            }
        }
        return lbs;
    }

    // Max
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Max>(expr)->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum_new(arg, parameters, assumptions, depth + 1, tight);
            if (lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (lbs == SymEngine::null) {
                lbs = lb;
            } else {
                lbs = symbolic::min(lbs, lb);
            }
        }
        return lbs;
    }

    // Min
    if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Min>(expr)->get_args();
        Expression lbs = SymEngine::null;
        for (const auto& arg : args) {
            auto lb = minimum_new(arg, parameters, assumptions, depth + 1, tight);
            if (lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (lbs == SymEngine::null) {
                lbs = lb;
            } else {
                lbs = symbolic::min(lbs, lb);
            }
        }
        return lbs;
    }

    return SymEngine::null;
}

Expression maximum_new(
    const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, const size_t depth, bool tight
) {
    // End of recursion: fail
    if (depth > MAX_DEPTH) {
        return SymEngine::null;
    }
    if (SymEngine::is_a<SymEngine::NaN>(*expr)) {
        return SymEngine::null;
    }
    if (SymEngine::is_a<SymEngine::Infty>(*expr)) {
        return SymEngine::null;
    }
    // End of recursion: success
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return expr;
    }
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (parameters.find(sym) != parameters.end()) {
            return sym;
        }
    }

    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        auto func_sym = SymEngine::rcp_static_cast<const SymEngine::FunctionSymbol>(expr);
        auto func_id = func_sym->get_name();
        if (func_id == "zext_i64") {
            auto zext = SymEngine::rcp_static_cast<const symbolic::ZExtI64Function>(expr);
            auto max_arg = maximum_new(zext->get_args()[0], parameters, assumptions, depth + 1, tight);
            if (max_arg == SymEngine::null) {
                return SymEngine::null;
            } else {
                return symbolic::zext_i64(max_arg);
            }
            if (max_arg == SymEngine::null) {
                return SymEngine::null;
            } else {
                return symbolic::zext_i64(max_arg);
            }
        } else if (func_id == "trunc_i32") {
            auto trunc = SymEngine::rcp_static_cast<const symbolic::TruncI32Function>(expr);
            auto max_arg = maximum_new(trunc->get_args()[0], parameters, assumptions, depth + 1, tight);
            if (max_arg == SymEngine::null) {
                return SymEngine::null;
            } else {
                return symbolic::trunc_i32(max_arg);
            }
        } else if (func_id == "idiv") {
            auto numerator = func_sym->get_args()[0];
            auto denominator = func_sym->get_args()[1];
            if (!SymEngine::is_a<const SymEngine::Integer>(*denominator)) {
                // Denominator is not a constant integer -> cannot soundly bound.
                return SymEngine::null;
            }

            auto numerator_ub = maximum_new(numerator, parameters, assumptions, depth + 1, tight);
            auto denominator_lb = minimum_new(denominator, parameters, assumptions, depth + 1, tight);
            if (numerator_ub == SymEngine::null || denominator_lb == SymEngine::null) {
                return SymEngine::null;
            }
            if (symbolic::is_true(symbolic::Le(denominator_lb, symbolic::zero()))) {
                // Denominator can be zero or negative -> cannot soundly bound.
                return SymEngine::null;
            }
            return symbolic::div(numerator_ub, denominator_lb);
        } else if (func_id == "imod") {
            auto lhs = func_sym->get_args()[0];
            auto rhs = func_sym->get_args()[1];
            if (!SymEngine::is_a<const SymEngine::Integer>(*rhs)) {
                return SymEngine::null;
            }

            auto lhs_lb = minimum_new(lhs, parameters, assumptions, depth + 1, tight);
            auto lhs_ub = maximum_new(lhs, parameters, assumptions, depth + 1, tight);
            if (lhs_lb == SymEngine::null || lhs_ub == SymEngine::null) {
                return SymEngine::null;
            }

            // Handle negative cases: max can be 0
            bool has_negative = symbolic::is_true(symbolic::Lt(lhs_ub, symbolic::zero())) ||
                                symbolic::is_true(symbolic::Lt(rhs, symbolic::zero()));
            auto pos_bound = symbolic::sub(rhs, symbolic::one());
            symbolic::Expression zero = symbolic::zero();

            auto width = symbolic::sub(lhs_ub, lhs_lb);
            if (symbolic::is_true(symbolic::Lt(width, rhs))) {
                // Range doesn't span full modulus cycle
                bool wraps = symbolic::is_true(symbolic::Lt(symbolic::mod(lhs_ub, rhs), symbolic::mod(lhs_lb, rhs)));
                if (wraps) {
                    return has_negative ? zero : pos_bound;
                }
                return symbolic::simplify(symbolic::mod(lhs_ub, rhs));
            }

            // Range spans full cycle
            return has_negative ? zero : pos_bound;
        }
    }

    // Symbol
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        auto sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(expr);
        if (assumptions.find(sym) != assumptions.end()) {
            if (tight) {
                if (assumptions.at(sym).tight_upper_bound().is_null()) {
                    return SymEngine::null;
                }
                return maximum_new(assumptions.at(sym).tight_upper_bound(), parameters, assumptions, depth + 1, tight);
            }
            symbolic::Expression new_ub = SymEngine::null;
            for (auto& ub : assumptions.at(sym).upper_bounds()) {
                auto new_max = maximum_new(ub, parameters, assumptions, depth + 1, tight);
                if (new_max.is_null()) {
                    continue;
                }
                if (new_ub.is_null()) {
                    new_ub = new_max;
                    continue;
                }
                new_ub = symbolic::min(new_ub, new_max);
            }
            return new_ub;
        }
        return SymEngine::null;
    }

    // Pow(base, k) with constant integer exponent k
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow_expr = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto args = pow_expr->get_args();
        if (args.size() != 2 || !SymEngine::is_a<SymEngine::Integer>(*args[1])) {
            return SymEngine::null;
        }

        long long exp_val = 0;
        try {
            exp_val = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
        } catch (const SymEngine::SymEngineException&) {
            return SymEngine::null;
        }

        if (exp_val < 0) {
            return SymEngine::null;
        }
        if (exp_val == 0) {
            return symbolic::one();
        }

        auto base_lb = minimum_new(args[0], parameters, assumptions, depth + 1, tight);
        auto base_ub = maximum_new(args[0], parameters, assumptions, depth + 1, tight);
        if (base_lb == SymEngine::null || base_ub == SymEngine::null) {
            return SymEngine::null;
        }

        auto exp_expr = symbolic::integer(exp_val);
        auto lb_pow = symbolic::pow(base_lb, exp_expr);
        auto ub_pow = symbolic::pow(base_ub, exp_expr);

        if (exp_val % 2 != 0) {
            return ub_pow;
        }

        return symbolic::max(lb_pow, ub_pow);
    }

    // Mul
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto mul = SymEngine::rcp_static_cast<const SymEngine::Mul>(expr);
        const auto& args = mul->get_args();
        size_t n = args.size();

        std::vector<std::pair<Expression, Expression>> bounds;
        bounds.reserve(n);

        for (const auto& arg : args) {
            Expression min_val = minimum_new(arg, parameters, assumptions, depth + 1, tight);
            Expression max_val = maximum_new(arg, parameters, assumptions, depth + 1, tight);

            if (min_val == SymEngine::null || max_val == SymEngine::null) {
                return SymEngine::null;
            }
            bounds.emplace_back(min_val, max_val);
        }

        // Iterate over 2^n combinations
        Expression max_product = SymEngine::null;
        const size_t total_combinations = 1ULL << n;

        for (size_t mask = 0; mask < total_combinations; ++mask) {
            Expression product = SymEngine::integer(1);
            for (size_t i = 0; i < n; ++i) {
                const auto& bound = bounds[i];
                Expression val = (mask & (1ULL << i)) ? bound.second : bound.first;
                product = symbolic::mul(product, val);
            }
            if (max_product == SymEngine::null) {
                max_product = product;
            } else {
                max_product = symbolic::max(max_product, product);
            }
        }

        return max_product;
    }

    // Add
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        const auto& args = add->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum_new(arg, parameters, assumptions, depth + 1, tight);
            if (ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (ubs == SymEngine::null) {
                ubs = ub;
            } else {
                ubs = symbolic::add(ubs, ub);
            }
        }
        return ubs;
    }

    // Max
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Max>(expr)->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum_new(arg, parameters, assumptions, depth + 1, tight);
            if (ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (ubs == SymEngine::null) {
                ubs = ub;
            } else {
                ubs = symbolic::max(ubs, ub);
            }
        }
        return ubs;
    }

    // Min
    if (SymEngine::is_a<SymEngine::Min>(*expr)) {
        auto args = SymEngine::rcp_dynamic_cast<const SymEngine::Min>(expr)->get_args();
        Expression ubs = SymEngine::null;
        for (const auto& arg : args) {
            auto ub = maximum_new(arg, parameters, assumptions, depth + 1, tight);
            if (ub == SymEngine::null) {
                return SymEngine::null;
            }
            if (ubs == SymEngine::null) {
                ubs = ub;
            } else {
                ubs = symbolic::max(ubs, ub);
            }
        }
        return ubs;
    }

    return SymEngine::null;
}

Expression minimum_new(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight) {
    return minimum_new(expr, parameters, assumptions, 0, tight);
}

Expression maximum_new(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight) {
    return maximum_new(expr, parameters, assumptions, 0, tight);
}

} // namespace symbolic
} // namespace sdfg
