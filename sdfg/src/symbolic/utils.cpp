#include "sdfg/symbolic/utils.h"

#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/ctx.h>
#include <isl/local_space.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/val.h>

#include <algorithm>
#include <unordered_set>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

// ISL-compatible printer. idiv/imod are C truncating `/` and `%`:
// trunc(a/b) = floord(max(a,0), b) - floord(max(-a,0), b) for b > 0, and for b < 0
// a / b == -(a / -b), a % b == a % -b. Non-literal divisors print as-is, which ISL rejects.
class ISLSymbolicPrinter : public SymEngine::BaseVisitor<ISLSymbolicPrinter, SymEngine::CodePrinter> {
public:
    using SymEngine::CodePrinter::apply;
    using SymEngine::CodePrinter::bvisit;
    using SymEngine::CodePrinter::str_;

    // CodePrinter emits C's `fmax`/`fmin` and `==`, which ISL rejects.
    void bvisit(const SymEngine::Max& x) {
        str_ = "max(" + join_args(x.get_args()) + ")";
    }
    void bvisit(const SymEngine::Min& x) {
        str_ = "min(" + join_args(x.get_args()) + ")";
    }
    void bvisit(const SymEngine::Equality& x) {
        str_ = apply(x.get_arg1()) + " = " + apply(x.get_arg2());
    }

    void bvisit(const SymEngine::FunctionSymbol& x) {
        const auto& args = x.get_args();
        if ((x.get_name() == "idiv" || x.get_name() == "imod") && args.size() == 2 &&
            SymEngine::is_a<SymEngine::Integer>(*args[1]) && !symbolic::eq(args[1], symbolic::zero())) {
            bool negative = SymEngine::down_cast<const SymEngine::Integer&>(*args[1]).is_negative();
            auto a = apply(args[0]);
            auto b = apply(negative ? symbolic::mul(symbolic::integer(-1), args[1]) : args[1]);
            auto quotient = "(floord(max((" + a + "), 0), " + b + ") - floord(max(-(" + a + "), 0), " + b + "))";
            if (x.get_name() == "imod") {
                // ISL rejects a parenthesized constant factor such as `(8)*(...)`.
                str_ = "((" + a + ") - " + b + "*" + quotient + ")";
            } else {
                str_ = negative ? "(-" + quotient + ")" : quotient;
            }
        } else if (x.get_name() == "iabs") {
            auto arg = apply(x.get_args()[0]);
            str_ = "max((" + arg + "), -(" + arg + "))";
        } else {
            // Unknown function - print as-is and let ISL reject it during parsing
            std::ostringstream ss;
            ss << x.get_name() << "(";
            auto args = x.get_args();
            for (size_t i = 0; i < args.size(); i++) {
                ss << apply(args[i]);
                if (i < args.size() - 1) {
                    ss << ", ";
                }
            }
            ss << ")";
            str_ = ss.str();
        }
    }

    void _print_pow(
        std::ostringstream& o,
        const SymEngine::RCP<const SymEngine::Basic>& a,
        const SymEngine::RCP<const SymEngine::Basic>& b
    ) {
        // ISL doesn't support power - expand to multiplication for positive integer exponents
        if (SymEngine::is_a<SymEngine::Integer>(*b)) {
            auto exp_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(b);
            try {
                long long val = exp_int->as_int();
                if (val >= 0 && val <= 10) { // Reasonable limit for expansion
                    std::string base_str = apply(a);
                    if (val == 0) {
                        o << "1";
                        return;
                    }
                    // For simple symbols/integers, don't add extra parentheses
                    bool needs_parens = !SymEngine::is_a<SymEngine::Symbol>(*a) &&
                                        !SymEngine::is_a<SymEngine::Integer>(*a);
                    for (long long i = 0; i < val; ++i) {
                        if (i > 0) {
                            o << "*";
                        }
                        if (needs_parens) {
                            o << "(" << base_str << ")";
                        } else {
                            o << base_str;
                        }
                    }
                    return;
                }
            } catch (const SymEngine::SymEngineException&) {
                // Fall through to default
            }
        }
        // Fall back to default printing (will likely fail in ISL)
        o << "(" << apply(a) << ")**(" << apply(b) << ")";
    }

private:
    std::string join_args(const SymEngine::vec_basic& args) {
        std::string s;
        for (size_t i = 0; i < args.size(); i++) {
            s += (i == 0 ? "" : ", ") + apply(args[i]);
        }
        return s;
    }
};

static ISLSymbolicPrinter isl_printer;

std::string expression_to_map_str(const MultiExpression& expr, const Assumptions& assums) {
    // Get all symbols
    symbolic::SymbolSet syms;
    for (auto& expr : expr) {
        auto syms_expr = symbolic::atoms(expr);
        syms.insert(syms_expr.begin(), syms_expr.end());
    }

    // Distinguish between dimensions and parameters
    std::vector<std::string> dimensions;
    SymbolSet dimensions_syms;
    std::vector<std::string> parameters;
    SymbolSet parameters_syms;
    for (auto& sym : syms) {
        auto it = assums.find(sym);
        if (it != assums.end() && it->second.constant() && it->second.map().is_null()) {
            if (parameters_syms.find(sym) != parameters_syms.end()) {
                continue;
            }
            parameters.push_back(sym->get_name());
            parameters_syms.insert(sym);
        } else {
            if (dimensions_syms.find(sym) != dimensions_syms.end()) {
                continue;
            }
            dimensions.push_back(sym->get_name());
            dimensions_syms.insert(sym);
        }
    }

    // Generate constraints
    SymbolSet seen;
    auto constraints_syms = generate_constraints(syms, assums, seen);

    // Extend parameters with additional symbols from constraints
    for (auto& con : constraints_syms) {
        auto con_syms = symbolic::atoms(con);
        for (auto& con_sym : con_syms) {
            if (dimensions_syms.find(con_sym) == dimensions_syms.end()) {
                if (parameters_syms.find(con_sym) != parameters_syms.end()) {
                    continue;
                }
                parameters.push_back(con_sym->get_name());
                parameters_syms.insert(con_sym);
            }
        }
    }

    // Define map
    std::stringstream map_ss;
    if (!parameters.empty()) {
        std::sort(parameters.begin(), parameters.end());
        map_ss << "[";
        map_ss << helpers::join(parameters, ", ");
        map_ss << "] -> ";
    }
    map_ss << "{ [" + helpers::join(dimensions, ", ") + "] -> [";
    for (size_t i = 0; i < expr.size(); i++) {
        auto dim = expr[i];
        map_ss << isl_printer.apply(dim);
        if (i < expr.size() - 1) {
            map_ss << ", ";
        }
    }
    map_ss << "] ";

    std::vector<std::string> constraints;
    for (auto& con : constraints_syms) {
        auto con_str = constraint_to_isl_str(con);
        if (!con_str.empty()) {
            constraints.push_back(con_str);
        }
    }
    for (auto& dim : dimensions) {
        auto sym = symbolic::symbol(dim);
        auto map_func = assums.at(sym).map();
        if (map_func == SymEngine::null) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Add>(*map_func)) {
            continue;
        }
        auto args = SymEngine::rcp_static_cast<const SymEngine::Add>(map_func)->get_args();
        if (args.size() != 2) {
            continue;
        }
        auto arg0 = args[0];
        auto arg1 = args[1];
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            arg0 = args[1];
            arg1 = args[0];
        }
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg1)) {
            continue;
        }
        if (symbolic::eq(arg1, symbolic::one())) {
            continue;
        }
        auto lb = assums.at(sym).tight_lower_bound();
        if (lb == SymEngine::null) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*lb)) {
            continue;
        }

        std::string iter = "__daisy_iterator_" + dim;
        std::string con = "exists " + iter + " : " + dim + " = " + isl_printer.apply(lb) + " + " + iter + " * " +
                          isl_printer.apply(arg1);
        constraints.push_back(con);
    }
    if (!constraints.empty()) {
        map_ss << " : ";
        map_ss << helpers::join(constraints, " and ");
    }

    map_ss << " }";

    std::string map = map_ss.str();
    return map;
}

namespace {

struct IntersectionSetup {
    std::vector<std::string> dimensions;
    SymbolSet dimensions_syms;
    std::vector<std::string> parameters;
    ExpressionSet constraints_syms_1;
    ExpressionSet constraints_syms_2;
};

IntersectionSetup intersection_setup(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
) {
    IntersectionSetup setup;
    auto& dimensions = setup.dimensions;
    auto& dimensions_syms = setup.dimensions_syms;
    auto& parameters = setup.parameters;
    auto& constraints_syms_1 = setup.constraints_syms_1;
    auto& constraints_syms_2 = setup.constraints_syms_2;

    // Get all symbols
    symbolic::SymbolSet syms;
    for (auto& expr : expr1) {
        auto syms_expr = symbolic::atoms(expr);
        syms.insert(syms_expr.begin(), syms_expr.end());
    }
    for (auto& expr : expr2) {
        auto syms_expr = symbolic::atoms(expr);
        syms.insert(syms_expr.begin(), syms_expr.end());
    }

    // Distinguish between dimensions and parameters
    SymbolSet parameters_syms;
    for (auto& sym : syms) {
        if (sym->get_name() != indvar->get_name() && assums1.at(sym).constant() && assums2.at(sym).constant()) {
            if (parameters_syms.find(sym) != parameters_syms.end()) {
                continue;
            }
            parameters.push_back(sym->get_name());
            parameters_syms.insert(sym);
        } else {
            if (dimensions_syms.find(sym) != dimensions_syms.end()) {
                continue;
            }
            dimensions.push_back(sym->get_name());
            dimensions_syms.insert(sym);
        }
    }

    // Generate constraints
    SymbolSet seen;
    constraints_syms_1 = generate_constraints(syms, assums1, seen);
    seen.clear();
    constraints_syms_2 = generate_constraints(syms, assums2, seen);

    // Extend parameters with additional symbols from constraints
    for (auto& con : constraints_syms_1) {
        auto con_syms = symbolic::atoms(con);
        for (auto& con_sym : con_syms) {
            if (dimensions_syms.find(con_sym) == dimensions_syms.end()) {
                if (parameters_syms.find(con_sym) != parameters_syms.end()) {
                    continue;
                }
                // Check if this symbol is the indvar - if so, it should be a dimension, not a parameter
                if (con_sym->get_name() == indvar->get_name()) {
                    dimensions.push_back(con_sym->get_name());
                    dimensions_syms.insert(con_sym);
                } else {
                    parameters.push_back(con_sym->get_name());
                    parameters_syms.insert(con_sym);
                }
            }
        }
    }
    for (auto& con : constraints_syms_2) {
        auto con_syms = symbolic::atoms(con);
        for (auto& con_sym : con_syms) {
            if (dimensions_syms.find(con_sym) == dimensions_syms.end()) {
                if (parameters_syms.find(con_sym) != parameters_syms.end()) {
                    continue;
                }
                // Check if this symbol is the indvar - if so, it should be a dimension, not a parameter
                if (con_sym->get_name() == indvar->get_name()) {
                    dimensions.push_back(con_sym->get_name());
                    dimensions_syms.insert(con_sym);
                } else {
                    parameters.push_back(con_sym->get_name());
                    parameters_syms.insert(con_sym);
                }
            }
        }
    }
    return setup;
}

} // namespace

std::tuple<std::string, std::string, std::string> expressions_to_intersection_map_str(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
) {
    auto setup = intersection_setup(expr1, expr2, indvar, assums1, assums2);
    const auto& dimensions = setup.dimensions;
    const auto& dimensions_syms = setup.dimensions_syms;
    auto& parameters = setup.parameters;
    const auto& constraints_syms_1 = setup.constraints_syms_1;
    const auto& constraints_syms_2 = setup.constraints_syms_2;

    // Define two maps
    std::stringstream map_1_ss;
    std::stringstream map_2_ss;
    if (!parameters.empty()) {
        map_1_ss << "[";
        map_1_ss << helpers::join(parameters, ", ");
        map_1_ss << "] -> ";
        map_2_ss << "[";
        map_2_ss << helpers::join(parameters, ", ");
        map_2_ss << "] -> ";
    }
    map_1_ss << "{ [";
    map_2_ss << "{ [";
    for (size_t i = 0; i < dimensions.size(); i++) {
        map_1_ss << dimensions[i] + "_1";
        map_2_ss << dimensions[i] + "_2";
        if (i < dimensions.size() - 1) {
            map_1_ss << ", ";
            map_2_ss << ", ";
        }
    }
    map_1_ss << "] -> [";
    map_2_ss << "] -> [";
    for (size_t i = 0; i < expr1.size(); i++) {
        auto dim = expr1[i];
        for (auto& iter : dimensions) {
            dim = symbolic::subs(dim, symbolic::symbol(iter), symbolic::symbol(iter + "_1"));
        }
        map_1_ss << isl_printer.apply(dim);
        if (i < expr1.size() - 1) {
            map_1_ss << ", ";
        }
    }
    for (size_t i = 0; i < expr2.size(); i++) {
        auto dim = expr2[i];
        for (auto& iter : dimensions) {
            dim = symbolic::subs(dim, symbolic::symbol(iter), symbolic::symbol(iter + "_2"));
        }
        map_2_ss << isl_printer.apply(dim);
        if (i < expr2.size() - 1) {
            map_2_ss << ", ";
        }
    }
    map_1_ss << "] ";
    map_2_ss << "] ";

    std::vector<std::string> constraints_1;
    // Add bounds
    for (auto& con : constraints_syms_1) {
        auto con_1 = con;
        for (auto& iter : dimensions) {
            con_1 = symbolic::subs(con_1, symbolic::symbol(iter), symbolic::symbol(iter + "_1"));
        }
        auto con_str_1 = constraint_to_isl_str(con_1);
        if (con_str_1.empty()) {
            continue;
        }
        constraints_1.push_back(con_str_1);
    }
    for (auto& dim : dimensions) {
        auto sym = symbolic::symbol(dim);
        auto map_func = assums1.at(sym).map();
        if (map_func == SymEngine::null) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Add>(*map_func)) {
            continue;
        }
        auto args = SymEngine::rcp_static_cast<const SymEngine::Add>(map_func)->get_args();
        if (args.size() != 2) {
            continue;
        }
        auto arg0 = args[0];
        auto arg1 = args[1];
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            arg0 = args[1];
            arg1 = args[0];
        }
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg1)) {
            continue;
        }
        if (symbolic::eq(arg1, symbolic::one())) {
            continue;
        }
        auto lb = assums1.at(sym).tight_lower_bound();
        if (!SymEngine::is_a<SymEngine::Integer>(*lb)) {
            continue;
        }

        std::string dim1 = dim + "_1";
        std::string iter = "__daisy_iterator_" + dim1;
        std::string con = "exists " + iter + " : " + dim1 + " = " + isl_printer.apply(lb) + " + " + iter + " * " +
                          isl_printer.apply(arg1);
        constraints_1.push_back(con);
    }
    if (!constraints_1.empty()) {
        map_1_ss << " : ";
        map_1_ss << helpers::join(constraints_1, " and ");
    }
    map_1_ss << " }";

    std::vector<std::string> constraints_2;
    for (auto& con : constraints_syms_2) {
        auto con_2 = con;
        for (auto& iter : dimensions) {
            con_2 = symbolic::subs(con_2, symbolic::symbol(iter), symbolic::symbol(iter + "_2"));
        }
        auto con_str_2 = constraint_to_isl_str(con_2);
        if (con_str_2.empty()) {
            continue;
        }
        constraints_2.push_back(con_str_2);
    }
    for (auto& dim : dimensions) {
        auto sym = symbolic::symbol(dim);
        auto map_func = assums2.at(sym).map();
        if (map_func.is_null()) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Add>(*map_func)) {
            continue;
        }
        auto args = SymEngine::rcp_static_cast<const SymEngine::Add>(map_func)->get_args();
        if (args.size() != 2) {
            continue;
        }
        auto arg0 = args[0];
        auto arg1 = args[1];
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            arg0 = args[1];
            arg1 = args[0];
        }
        if (!symbolic::eq(arg0, symbolic::symbol(dim))) {
            continue;
        }
        if (!SymEngine::is_a<SymEngine::Integer>(*arg1)) {
            continue;
        }
        if (symbolic::eq(arg1, symbolic::one())) {
            continue;
        }
        auto lb = assums2.at(sym).tight_lower_bound();
        if (!SymEngine::is_a<SymEngine::Integer>(*lb)) {
            continue;
        }

        std::string dim2 = dim + "_2";
        std::string iter = "__daisy_iterator_" + dim2;
        std::string con = "exists " + iter + " : " + dim2 + " = " + isl_printer.apply(lb) + " + " + iter + " * " +
                          isl_printer.apply(arg1);
        constraints_2.push_back(con);
    }
    if (!constraints_2.empty()) {
        map_2_ss << " : ";
        map_2_ss << helpers::join(constraints_2, " and ");
    }
    map_2_ss << " }";

    std::stringstream map_3_ss;
    map_3_ss << "{ [";
    for (size_t i = 0; i < dimensions.size(); i++) {
        map_3_ss << dimensions[i] + "_2";
        if (i < dimensions.size() - 1) {
            map_3_ss << ", ";
        }
    }
    map_3_ss << "] -> [";
    for (size_t i = 0; i < dimensions.size(); i++) {
        map_3_ss << dimensions[i] + "_1";
        if (i < dimensions.size() - 1) {
            map_3_ss << ", ";
        }
    }
    map_3_ss << "]";
    std::vector<std::string> monotonicity_constraints;
    if (dimensions_syms.find(indvar) != dimensions_syms.end()) {
        // For loop-carried dependencies, we only care about the forward direction
        // (iteration_1 < iteration_2 in execution order).
        // This ensures we detect real dependencies, not symmetric pairs.
        monotonicity_constraints.push_back(indvar->get_name() + "_1 < " + indvar->get_name() + "_2");
    }
    if (!monotonicity_constraints.empty()) {
        map_3_ss << " : ";
        map_3_ss << helpers::join(monotonicity_constraints, " and ");
    }
    map_3_ss << " }";

    std::string map_1 = map_1_ss.str();
    std::string map_2 = map_2_ss.str();
    std::string map_3 = map_3_ss.str();

    return {map_1, map_2, map_3};
}

namespace {

using IslNameTable = std::unordered_map<std::string, std::pair<isl_dim_type, unsigned>>;

isl_val* isl_val_from_integer(isl_ctx* ctx, const SymEngine::Integer& x) {
    if (mp_fits_slong_p(x.as_integer_class())) {
        return isl_val_int_from_si(ctx, mp_get_si(x.as_integer_class()));
    }
    return isl_val_read_from_str(ctx, x.__str__().c_str());
}

// Literal divisor, as required by ISL's quotient/remainder (division by zero is undefined).
bool is_nonzero_integer(const Expression& e) {
    return SymEngine::is_a<SymEngine::Integer>(*e) && !symbolic::eq(e, symbolic::zero());
}

// Returns null for non-affine products, rationals, powers, unknown functions or names.
isl_pw_aff* expression_to_isl_pw_aff(const Expression& e, isl_local_space* ls, const IslNameTable& names) {
    isl_ctx* ctx = isl_local_space_get_ctx(ls);
    if (SymEngine::is_a<SymEngine::Integer>(*e)) {
        return isl_pw_aff_from_aff(isl_aff_val_on_domain(
            isl_local_space_copy(ls), isl_val_from_integer(ctx, SymEngine::down_cast<const SymEngine::Integer&>(*e))
        ));
    }
    if (SymEngine::is_a<SymEngine::Symbol>(*e)) {
        auto it = names.find(SymEngine::down_cast<const SymEngine::Symbol&>(*e).get_name());
        if (it == names.end()) {
            return nullptr;
        }
        return isl_pw_aff_from_aff(isl_aff_var_on_domain(isl_local_space_copy(ls), it->second.first, it->second.second));
    }
    bool is_add = SymEngine::is_a<SymEngine::Add>(*e);
    bool is_max = SymEngine::is_a<SymEngine::Max>(*e);
    bool is_min = SymEngine::is_a<SymEngine::Min>(*e);
    if (is_add || is_max || is_min) {
        isl_pw_aff* acc = nullptr;
        for (auto& arg : e->get_args()) {
            isl_pw_aff* term = expression_to_isl_pw_aff(arg, ls, names);
            if (!term) {
                isl_pw_aff_free(acc);
                return nullptr;
            }
            if (!acc) {
                acc = term;
            } else if (is_add) {
                acc = isl_pw_aff_add(acc, term);
            } else if (is_max) {
                acc = isl_pw_aff_max(acc, term);
            } else {
                acc = isl_pw_aff_min(acc, term);
            }
        }
        return acc;
    }
    if (SymEngine::is_a<SymEngine::Mul>(*e)) {
        auto& mul = SymEngine::down_cast<const SymEngine::Mul&>(*e);
        auto& dict = mul.get_dict();
        if (!SymEngine::is_a<SymEngine::Integer>(*mul.get_coef()) || dict.size() != 1 ||
            !symbolic::eq(dict.begin()->second, symbolic::one())) {
            return nullptr;
        }
        isl_pw_aff* base = expression_to_isl_pw_aff(dict.begin()->first, ls, names);
        if (!base) {
            return nullptr;
        }
        return isl_pw_aff_scale_val(
            base, isl_val_from_integer(ctx, SymEngine::down_cast<const SymEngine::Integer&>(*mul.get_coef()))
        );
    }
    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*e)) {
        auto& func = SymEngine::down_cast<const SymEngine::FunctionSymbol&>(*e);
        auto args = func.get_args();
        const auto& name = func.get_name();
        if (name == "iabs" && args.size() == 1) {
            isl_pw_aff* arg = expression_to_isl_pw_aff(args[0], ls, names);
            if (!arg) {
                return nullptr;
            }
            // Copy before neg: argument evaluation order is unspecified and neg consumes `arg`.
            isl_pw_aff* neg = isl_pw_aff_neg(isl_pw_aff_copy(arg));
            return isl_pw_aff_max(arg, neg);
        }
        if ((name == "idiv" || name == "imod") && args.size() == 2 && is_nonzero_integer(args[1])) {
            isl_pw_aff* arg = expression_to_isl_pw_aff(args[0], ls, names);
            if (!arg) {
                return nullptr;
            }
            // C `/` and `%` truncate: a / b == -(a / |b|) and a % b == a % |b| for b < 0.
            const auto& b = SymEngine::down_cast<const SymEngine::Integer&>(*args[1]);
            bool negative = b.is_negative();
            isl_val* magnitude = isl_val_abs(isl_val_from_integer(ctx, b));
            isl_pw_aff* divisor = isl_pw_aff_from_aff(isl_aff_val_on_domain(isl_local_space_copy(ls), magnitude));
            if (name == "imod") {
                return isl_pw_aff_tdiv_r(arg, divisor);
            }
            isl_pw_aff* quotient = isl_pw_aff_tdiv_q(arg, divisor);
            return negative ? isl_pw_aff_neg(quotient) : quotient;
        }
    }
    return nullptr;
}

// Builds `{ [dims] -> [exprs] : constraints and strides }` like the corresponding parsed string.
isl_map* build_access_map(
    isl_ctx* ctx,
    const IntersectionSetup& setup,
    const MultiExpression& exprs,
    const ExpressionSet& constraints,
    const Assumptions& assums,
    const std::string& suffix
) {
    const size_t n_params = setup.parameters.size();
    const size_t n_dims = setup.dimensions.size();

    isl_space* domain = isl_space_set_alloc(ctx, n_params, n_dims);
    IslNameTable names;
    for (size_t i = 0; i < n_params; i++) {
        domain = isl_space_set_dim_name(domain, isl_dim_param, i, setup.parameters[i].c_str());
        names.emplace(setup.parameters[i], std::make_pair(isl_dim_param, unsigned(i)));
    }
    for (size_t i = 0; i < n_dims; i++) {
        domain = isl_space_set_dim_name(domain, isl_dim_set, i, (setup.dimensions[i] + suffix).c_str());
        // Keyed by the unsuffixed name: the string path renames dims inside expressions.
        names.emplace(setup.dimensions[i], std::make_pair(isl_dim_set, unsigned(i)));
    }
    isl_local_space* ls = isl_local_space_from_space(isl_space_copy(domain));

    isl_map* result = nullptr;
    isl_basic_set* bset = isl_basic_set_universe(isl_space_copy(domain));
    // Constraints that are not a single basic set (`!=`, piecewise Max/Min operands).
    isl_set* extra_sets = nullptr;
    isl_pw_aff_list* outs = isl_pw_aff_list_alloc(ctx, exprs.size());
    bool outs_affine = true;
    bool ok = true;

    for (auto& e : exprs) {
        isl_pw_aff* pa = expression_to_isl_pw_aff(e, ls, names);
        if (!pa) {
            ok = false;
            break;
        }
        outs_affine = outs_affine && isl_pw_aff_isa_aff(pa) == isl_bool_true;
        outs = isl_pw_aff_list_add(outs, pa);
    }

    for (auto it = constraints.begin(); ok && it != constraints.end(); ++it) {
        auto& con = *it;
        bool is_lt = SymEngine::is_a<SymEngine::StrictLessThan>(*con);
        bool is_le = SymEngine::is_a<SymEngine::LessThan>(*con);
        bool is_ne = SymEngine::is_a<SymEngine::Unequality>(*con);
        bool is_eq = SymEngine::is_a<SymEngine::Equality>(*con);
        if (!is_lt && !is_le && !is_ne && !is_eq) {
            continue;
        }
        auto args = con->get_args();
        if (SymEngine::is_a<SymEngine::Infty>(*args[0]) || SymEngine::is_a<SymEngine::Infty>(*args[1])) {
            continue;
        }
        isl_pw_aff* lhs = expression_to_isl_pw_aff(args[0], ls, names);
        isl_pw_aff* rhs = expression_to_isl_pw_aff(args[1], ls, names);
        if (!lhs || !rhs) {
            isl_pw_aff_free(lhs);
            isl_pw_aff_free(rhs);
            ok = false;
            break;
        }
        isl_set* s = nullptr;
        if (is_ne) {
            s = isl_pw_aff_ne_set(lhs, rhs);
        } else if (isl_pw_aff_isa_aff(lhs) == isl_bool_true && isl_pw_aff_isa_aff(rhs) == isl_bool_true) {
            isl_aff* l = isl_pw_aff_as_aff(lhs);
            isl_aff* r = isl_pw_aff_as_aff(rhs);
            isl_basic_set* c = is_eq   ? isl_aff_eq_basic_set(l, r)
                               : is_lt ? isl_aff_lt_basic_set(l, r)
                                       : isl_aff_le_basic_set(l, r);
            bset = isl_basic_set_intersect(bset, c);
        } else {
            s = is_eq ? isl_pw_aff_eq_set(lhs, rhs) : is_lt ? isl_pw_aff_lt_set(lhs, rhs) : isl_pw_aff_le_set(lhs, rhs);
        }
        if (s) {
            extra_sets = extra_sets ? isl_set_intersect(extra_sets, s) : s;
        }
    }

    // `exists it : dim = lb + it * step`, emitted for constant non-unit strides.
    for (size_t i = 0; ok && i < n_dims; i++) {
        auto sym = symbolic::symbol(setup.dimensions[i]);
        auto map_func = assums.at(sym).map();
        if (map_func.is_null() || !SymEngine::is_a<SymEngine::Add>(*map_func)) {
            continue;
        }
        auto args = SymEngine::rcp_static_cast<const SymEngine::Add>(map_func)->get_args();
        if (args.size() != 2) {
            continue;
        }
        auto arg0 = args[0];
        auto arg1 = args[1];
        if (!symbolic::eq(arg0, sym)) {
            arg0 = args[1];
            arg1 = args[0];
        }
        if (!symbolic::eq(arg0, sym) || !SymEngine::is_a<SymEngine::Integer>(*arg1) ||
            symbolic::eq(arg1, symbolic::one())) {
            continue;
        }
        auto lb = assums.at(sym).tight_lower_bound();
        if (lb.is_null() || !SymEngine::is_a<SymEngine::Integer>(*lb)) {
            continue;
        }
        isl_aff* offset = isl_aff_sub(
            isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, i),
            isl_aff_val_on_domain(
                isl_local_space_copy(ls),
                isl_val_from_integer(ctx, SymEngine::down_cast<const SymEngine::Integer&>(*lb))
            )
        );
        isl_val* step = isl_val_abs(isl_val_from_integer(ctx, SymEngine::down_cast<const SymEngine::Integer&>(*arg1)));
        if (!isl_val_is_zero(step)) {
            offset = isl_aff_mod_val(offset, step);
        } else {
            isl_val_free(step);
        }
        bset = isl_basic_set_intersect(bset, isl_aff_zero_basic_set(offset));
    }

    if (ok) {
        isl_space* range = isl_space_add_dims(
            isl_space_set_from_params(isl_space_params(isl_space_copy(domain))), isl_dim_set, exprs.size()
        );
        isl_space* space = isl_space_map_from_domain_and_range(isl_space_copy(domain), range);
        isl_set* dom = isl_set_from_basic_set(bset);
        bset = nullptr;
        if (extra_sets) {
            dom = isl_set_intersect(dom, extra_sets);
            extra_sets = nullptr;
        }
        isl_map* access;
        if (outs_affine) {
            isl_aff_list* affs = isl_aff_list_alloc(ctx, exprs.size());
            for (int i = 0; i < isl_pw_aff_list_n_pw_aff(outs); i++) {
                affs = isl_aff_list_add(affs, isl_pw_aff_as_aff(isl_pw_aff_list_get_pw_aff(outs, i)));
            }
            access = isl_map_from_multi_aff(isl_multi_aff_from_aff_list(space, affs));
        } else {
            access = isl_map_from_multi_pw_aff(isl_multi_pw_aff_from_pw_aff_list(space, isl_pw_aff_list_copy(outs)));
        }
        result = isl_map_intersect_domain(access, dom);
    }

    isl_pw_aff_list_free(outs);
    isl_basic_set_free(bset);
    isl_set_free(extra_sets);
    isl_local_space_free(ls);
    isl_space_free(domain);
    return result;
}

} // namespace

bool expressions_to_intersection_maps(
    isl_ctx* ctx,
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2,
    isl_map** map_1,
    isl_map** map_2,
    isl_map** map_3
) {
    *map_1 = nullptr;
    *map_2 = nullptr;
    if (map_3) {
        *map_3 = nullptr;
    }
    auto setup = intersection_setup(expr1, expr2, indvar, assums1, assums2);

    // Name clashes make the string path's sequential renaming and name lookup ambiguous.
    std::unordered_set<std::string> taken(setup.parameters.begin(), setup.parameters.end());
    taken.insert(setup.dimensions.begin(), setup.dimensions.end());
    if (taken.size() != setup.parameters.size() + setup.dimensions.size()) {
        return false;
    }
    for (auto& dim : setup.dimensions) {
        if (taken.count(dim + "_1") || taken.count(dim + "_2")) {
            return false;
        }
    }

    *map_1 = build_access_map(ctx, setup, expr1, setup.constraints_syms_1, assums1, "_1");
    *map_2 = *map_1 ? build_access_map(ctx, setup, expr2, setup.constraints_syms_2, assums2, "_2") : nullptr;
    if (!*map_1 || !*map_2) {
        *map_1 = isl_map_free(*map_1);
        *map_2 = isl_map_free(*map_2);
        return false;
    }
    if (!map_3) {
        return true;
    }

    // `{ [dims_2] -> [dims_1] : indvar_1 < indvar_2 }`
    const size_t n_dims = setup.dimensions.size();
    isl_space* space = isl_space_alloc(ctx, 0, n_dims, n_dims);
    for (size_t i = 0; i < n_dims; i++) {
        space = isl_space_set_dim_name(space, isl_dim_in, i, (setup.dimensions[i] + "_2").c_str());
        space = isl_space_set_dim_name(space, isl_dim_out, i, (setup.dimensions[i] + "_1").c_str());
    }
    isl_map* mono = isl_map_universe(isl_space_copy(space));
    auto pos = std::find(setup.dimensions.begin(), setup.dimensions.end(), indvar->get_name());
    if (setup.dimensions_syms.count(indvar) && pos != setup.dimensions.end()) {
        unsigned p = pos - setup.dimensions.begin();
        isl_constraint* c = isl_constraint_alloc_inequality(isl_local_space_from_space(isl_space_copy(space)));
        c = isl_constraint_set_coefficient_si(c, isl_dim_in, p, 1);
        c = isl_constraint_set_coefficient_si(c, isl_dim_out, p, -1);
        c = isl_constraint_set_constant_si(c, -1);
        mono = isl_map_add_constraint(mono, c);
    }
    isl_space_free(space);
    *map_3 = mono;
    if (!mono) {
        *map_1 = isl_map_free(*map_1);
        *map_2 = isl_map_free(*map_2);
        return false;
    }
    return true;
}

ExpressionSet generate_constraints(SymbolSet& syms, const Assumptions& assums, SymbolSet& seen) {
    ExpressionSet constraints;
    for (auto& sym : syms) {
        if (assums.find(sym) == assums.end()) {
            continue;
        }
        if (seen.find(sym) != seen.end()) {
            continue;
        }
        seen.insert(sym);

        for (auto& ub : assums.at(sym).upper_bounds()) {
            auto con = symbolic::Le(sym, ub);
            auto con_syms = symbolic::atoms(con);
            constraints.insert(con);

            auto con_cons = generate_constraints(con_syms, assums, seen);
            constraints.insert(con_cons.begin(), con_cons.end());
        }
        for (auto& lb : assums.at(sym).lower_bounds()) {
            auto con = symbolic::Ge(sym, lb);
            auto con_syms = symbolic::atoms(con);
            constraints.insert(con);

            auto con_cons = generate_constraints(con_syms, assums, seen);
            constraints.insert(con_cons.begin(), con_cons.end());
        }
    }
    return constraints;
}

std::string constraint_to_isl_str(const Expression con) {
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*con)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(con);
        auto lhs = le->get_arg1();
        auto rhs = le->get_arg2();
        if (SymEngine::is_a<SymEngine::Infty>(*lhs) || SymEngine::is_a<SymEngine::Infty>(*rhs)) {
            return "";
        }
        auto res = isl_printer.apply(con);
        return res;
    } else if (SymEngine::is_a<SymEngine::LessThan>(*con)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(con);
        auto lhs = le->get_arg1();
        auto rhs = le->get_arg2();
        if (SymEngine::is_a<SymEngine::Infty>(*lhs) || SymEngine::is_a<SymEngine::Infty>(*rhs)) {
            return "";
        }
        auto res = isl_printer.apply(con);
        return res;
    } else if (SymEngine::is_a<SymEngine::Equality>(*con)) {
        auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(con);
        auto lhs = eq->get_arg1();
        auto rhs = eq->get_arg2();
        if (SymEngine::is_a<SymEngine::Infty>(*lhs) || SymEngine::is_a<SymEngine::Infty>(*rhs)) {
            return "";
        }
        auto res = isl_printer.apply(con);
        return res;
    } else if (SymEngine::is_a<SymEngine::Unequality>(*con)) {
        auto ne = SymEngine::rcp_static_cast<const SymEngine::Unequality>(con);
        auto lhs = ne->get_arg1();
        auto rhs = ne->get_arg2();
        if (SymEngine::is_a<SymEngine::Infty>(*lhs) || SymEngine::is_a<SymEngine::Infty>(*rhs)) {
            return "";
        }
        auto res = isl_printer.apply(con);
        return res;
    }

    return "";
}

void canonicalize_map_dims(isl_map* map, const std::string& in_prefix, const std::string& out_prefix) {
    int n_in = isl_map_dim(map, isl_dim_in);
    int n_out = isl_map_dim(map, isl_dim_out);

    for (int i = 0; i < n_in; ++i) {
        std::string name = in_prefix + std::to_string(i);
        map = isl_map_set_dim_name(map, isl_dim_in, i, name.c_str());
    }

    for (int i = 0; i < n_out; ++i) {
        std::string name = out_prefix + std::to_string(i);
        map = isl_map_set_dim_name(map, isl_dim_out, i, name.c_str());
    }
}

bool vectors_of_expressions_match(const std::vector<Expression>& a, const std::vector<Expression>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!symbolic::eq(a.at(i), b.at(i))) {
            return false;
        }
    }
    return true;
}

bool vectors_of_expressions_match(
    const std::vector<Expression>& a, const std::vector<Expression>& b, const ExpressionMapping& replacements
) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!symbolic::eq(SymEngine::subs(a.at(i), replacements), b.at(i))) {
            return false;
        }
    }
    return true;
}

} // namespace symbolic
} // namespace sdfg
