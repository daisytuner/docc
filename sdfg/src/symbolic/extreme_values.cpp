#include "sdfg/symbolic/extreme_values.h"

#include "sdfg/symbolic/polynomials.h"

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

    // Pow with constant exponent
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto base = pow->get_base();
        auto exp = pow->get_exp();
        if (SymEngine::is_a<SymEngine::Integer>(*exp)) {
            auto exp_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(exp)->as_int();
            if (exp_int >= 0) {
                auto min_base = minimum(base, parameters, assumptions, depth + 1);
                if (min_base == SymEngine::null) {
                    return SymEngine::null;
                }
                symbolic::Expression min_expr = symbolic::one();
                for (int i = 0; i < exp_int; ++i) {
                    min_expr = symbolic::mul(min_expr, min_base);
                }
                return min_expr;
            }
        }
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

    // Pow with constant exponent
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto base = pow->get_base();
        auto exp = pow->get_exp();
        if (SymEngine::is_a<SymEngine::Integer>(*exp)) {
            auto exp_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(exp)->as_int();
            if (exp_int >= 0) {
                auto max_base = maximum(base, parameters, assumptions, depth + 1);
                if (max_base == SymEngine::null) {
                    return SymEngine::null;
                }
                symbolic::Expression max_expr = symbolic::one();
                for (int i = 0; i < exp_int; ++i) {
                    max_expr = symbolic::mul(max_expr, max_base);
                }
                return max_expr;
            }
        }
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

    if (SymEngine::is_a<symbolic::ZExtI64Function>(*expr)) {
        auto zext = SymEngine::rcp_static_cast<const symbolic::ZExtI64Function>(expr);
        auto min_arg = minimum_new(zext->get_args()[0], parameters, assumptions, depth + 1, tight);
        if (min_arg == SymEngine::null) {
            return SymEngine::null;
        } else {
            return symbolic::zext_i64(min_arg);
        }
    }
    if (SymEngine::is_a<symbolic::TruncI32Function>(*expr)) {
        auto trunc = SymEngine::rcp_static_cast<const symbolic::TruncI32Function>(expr);
        auto min_arg = minimum_new(trunc->get_args()[0], parameters, assumptions, depth + 1, tight);
        if (min_arg == SymEngine::null) {
            return SymEngine::null;
        } else {
            return symbolic::trunc_i32(min_arg);
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

    // Pow with constant exponent
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto base = pow->get_base();
        auto exp = pow->get_exp();
        if (SymEngine::is_a<SymEngine::Integer>(*exp)) {
            auto exp_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(exp)->as_int();
            if (exp_int >= 0) {
                auto min_base = minimum_new(base, parameters, assumptions, depth + 1, tight);
                if (min_base == SymEngine::null) {
                    return SymEngine::null;
                }
                symbolic::Expression min_expr = symbolic::one();
                for (int i = 0; i < exp_int; ++i) {
                    min_expr = symbolic::mul(min_expr, min_base);
                }
                return min_expr;
            }
        }
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

    if (SymEngine::is_a<symbolic::ZExtI64Function>(*expr)) {
        auto zext = SymEngine::rcp_static_cast<const symbolic::ZExtI64Function>(expr);
        auto max_arg = maximum_new(zext->get_args()[0], parameters, assumptions, depth + 1, tight);
        if (max_arg == SymEngine::null) {
            return SymEngine::null;
        } else {
            return symbolic::zext_i64(max_arg);
        }
    }
    if (SymEngine::is_a<symbolic::TruncI32Function>(*expr)) {
        auto trunc = SymEngine::rcp_static_cast<const symbolic::TruncI32Function>(expr);
        auto max_arg = maximum_new(trunc->get_args()[0], parameters, assumptions, depth + 1, tight);
        if (max_arg == SymEngine::null) {
            return SymEngine::null;
        } else {
            return symbolic::trunc_i32(max_arg);
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

    // Pow with constant exponent
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto base = pow->get_base();
        auto exp = pow->get_exp();
        if (SymEngine::is_a<SymEngine::Integer>(*exp)) {
            auto exp_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(exp)->as_int();
            if (exp_int >= 0) {
                auto max_base = maximum_new(base, parameters, assumptions, depth + 1, tight);
                if (max_base == SymEngine::null) {
                    return SymEngine::null;
                }
                symbolic::Expression max_expr = symbolic::one();
                for (int i = 0; i < exp_int; ++i) {
                    max_expr = symbolic::mul(max_expr, max_base);
                }
                return max_expr;
            }
        }
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
