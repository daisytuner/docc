#include "sdfg/symbolic/symbolic.h"

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
        auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
        auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
        return integer(a / b);
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
                auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
                auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
                return integer(a / b);
            }
        } else if (func_id == "imod") {
            auto lhs = func_sym->get_args()[0];
            auto rhs = func_sym->get_args()[1];
            if (SymEngine::is_a<SymEngine::Integer>(*lhs) && SymEngine::is_a<SymEngine::Integer>(*rhs)) {
                auto a = SymEngine::rcp_static_cast<const SymEngine::Integer>(lhs)->as_int();
                auto b = SymEngine::rcp_static_cast<const SymEngine::Integer>(rhs)->as_int();
                return integer(a % b);
            }
        } else if (func_id == "zext_i64") {
            auto arg = func_sym->get_args()[0];
            auto simple_arg = symbolic::simplify(arg);

            bool non_negative = false;
            if (SymEngine::is_a<SymEngine::Integer>(*simple_arg)) {
                if (SymEngine::rcp_static_cast<const SymEngine::Integer>(simple_arg)->as_int() >= 0) {
                    non_negative = true;
                }
            } else if (SymEngine::is_a<SymEngine::Max>(*simple_arg)) {
                auto max_op = SymEngine::rcp_static_cast<const SymEngine::Max>(simple_arg);
                for (const auto& m_arg : max_op->get_args()) {
                    if (SymEngine::is_a<SymEngine::Integer>(*m_arg)) {
                        if (SymEngine::rcp_static_cast<const SymEngine::Integer>(m_arg)->as_int() >= 0) {
                            non_negative = true;
                            break;
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
                auto val = SymEngine::rcp_static_cast<const SymEngine::Integer>(simple_arg)->as_int();
                return integer(val >= 0 ? val : -val);
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
    AffineCoeffs affine_coeffs = affine_coefficients(poly, symbols);
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
