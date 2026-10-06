#include "sdfg/symbolic/utils.h"

#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/ctx.h>
#include <isl/local_space.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/val.h>
#include <symengine/ntheory.h>

#include <algorithm>
#include <unordered_set>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polyhedral.h"
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

// Integer residue of a strided loop's lower bound: `c + step * k` (k integer-valued) yields `c`, else null.
static Expression stride_residue(const Expression& lb, const Expression& step) {
    if (lb.is_null() || !SymEngine::is_a<SymEngine::Integer>(*step)) {
        return SymEngine::null;
    }
    if (SymEngine::is_a<SymEngine::Integer>(*lb)) {
        return lb;
    }
    auto& step_int = SymEngine::down_cast<const SymEngine::Integer&>(*step);
    auto multiple_of_step = [&](const Expression& term) {
        if (!SymEngine::is_a<SymEngine::Mul>(*term)) {
            return false;
        }
        auto coef = SymEngine::down_cast<const SymEngine::Mul&>(*term).get_coef();
        return SymEngine::is_a<SymEngine::Integer>(*coef) &&
               SymEngine::mp_divisible_p(
                   SymEngine::down_cast<const SymEngine::Integer&>(*coef).as_integer_class(),
                   step_int.as_integer_class()
               );
    };
    auto expanded = symbolic::expand(lb);
    if (multiple_of_step(expanded)) {
        return symbolic::zero();
    }
    if (!SymEngine::is_a<SymEngine::Add>(*expanded)) {
        return SymEngine::null;
    }
    Expression residue = symbolic::zero();
    for (auto& term : expanded->get_args()) {
        if (SymEngine::is_a<SymEngine::Integer>(*term)) {
            residue = symbolic::add(residue, term);
        } else if (!multiple_of_step(term)) {
            return SymEngine::null;
        }
    }
    return residue;
}

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
        auto lb = stride_residue(assums.at(sym).tight_lower_bound(), arg1);
        if (lb == SymEngine::null) {
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
        auto lb = stride_residue(assums1.at(sym).tight_lower_bound(), arg1);
        if (lb.is_null()) {
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
        auto lb = stride_residue(assums2.at(sym).tight_lower_bound(), arg1);
        if (lb.is_null()) {
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

// Floor division by a positive literal; only produced by split_extremum for the isl conversion below.
const char* const floordiv_name = "__isl_floordiv";

// Literal divisor, as required by ISL's quotient/remainder (division by zero is undefined).
bool is_nonzero_integer(const Expression& e) {
    return SymEngine::is_a<SymEngine::Integer>(*e) && !symbolic::eq(e, symbolic::zero());
}

// Whether `pa` (not consumed) is >= 0 everywhere in `context` (null: unknown).
bool nonneg_in(isl_pw_aff* pa, isl_basic_set* context) {
    if (!context) {
        return false;
    }
    if (isl_pw_aff_isa_aff(pa) == isl_bool_true) {
        // Basic sets skip the costly normalization isl_set_intersect does to detect equal operands.
        isl_basic_set* neg = isl_aff_neg_basic_set(isl_pw_aff_as_aff(isl_pw_aff_copy(pa)));
        neg = isl_basic_set_intersect(neg, isl_basic_set_copy(context));
        bool empty = isl_basic_set_is_empty(neg) == isl_bool_true;
        isl_basic_set_free(neg);
        return empty;
    }
    isl_pw_aff* zero = isl_pw_aff_zero_on_domain(isl_local_space_from_space(isl_pw_aff_get_domain_space(pa)));
    isl_set* neg = isl_pw_aff_lt_set(isl_pw_aff_copy(pa), zero);
    neg = isl_set_intersect(neg, isl_set_from_basic_set(isl_basic_set_copy(context)));
    bool empty = isl_set_is_empty(neg) == isl_bool_true;
    isl_set_free(neg);
    return empty;
}

// `e` as a quasi-affine isl_aff (C division only where the numerator is non-negative in `context`), or null.
isl_aff* expression_to_isl_aff(const Expression& e, isl_local_space* ls, const IslNameTable& names, isl_basic_set* context) {
    isl_ctx* ctx = isl_local_space_get_ctx(ls);
    if (SymEngine::is_a<SymEngine::Integer>(*e)) {
        return isl_aff_val_on_domain(
            isl_local_space_copy(ls), isl_val_from_integer(ctx, SymEngine::down_cast<const SymEngine::Integer&>(*e))
        );
    }
    if (SymEngine::is_a<SymEngine::Symbol>(*e)) {
        auto it = names.find(SymEngine::down_cast<const SymEngine::Symbol&>(*e).get_name());
        if (it == names.end()) {
            return nullptr;
        }
        return isl_aff_var_on_domain(isl_local_space_copy(ls), it->second.first, it->second.second);
    }
    if (SymEngine::is_a<SymEngine::Add>(*e)) {
        // Walks the dict: Add::get_args allocates a Mul per term.
        auto& add = SymEngine::down_cast<const SymEngine::Add&>(*e);
        if (!SymEngine::is_a<SymEngine::Integer>(*add.get_coef())) {
            return nullptr;
        }
        isl_aff* acc = expression_to_isl_aff(add.get_coef(), ls, names, context);
        for (auto& [term, coef] : add.get_dict()) {
            isl_aff* t = SymEngine::is_a<SymEngine::Integer>(*coef) ? expression_to_isl_aff(term, ls, names, context)
                                                                    : nullptr;
            if (!t) {
                isl_aff_free(acc);
                return nullptr;
            }
            t = isl_aff_scale_val(t, isl_val_from_integer(ctx, SymEngine::down_cast<const SymEngine::Integer&>(*coef)));
            acc = isl_aff_add(acc, t);
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
        isl_aff* base = expression_to_isl_aff(dict.begin()->first, ls, names, context);
        if (!base) {
            return nullptr;
        }
        return isl_aff_scale_val(
            base, isl_val_from_integer(ctx, SymEngine::down_cast<const SymEngine::Integer&>(*mul.get_coef()))
        );
    }
    if (!SymEngine::is_a<SymEngine::FunctionSymbol>(*e)) {
        return nullptr;
    }
    auto& func = SymEngine::down_cast<const SymEngine::FunctionSymbol&>(*e);
    auto args = func.get_args();
    const auto& name = func.get_name();
    bool c_division = name == "idiv" || name == "imod";
    if ((!c_division && name != floordiv_name) || args.size() != 2 || !is_nonzero_integer(args[1])) {
        return nullptr;
    }
    isl_aff* arg = expression_to_isl_aff(args[0], ls, names, context);
    if (!arg) {
        return nullptr;
    }
    const auto& b = SymEngine::down_cast<const SymEngine::Integer&>(*args[1]);
    if (c_division) {
        bool nonneg = false;
        if (context) {
            isl_basic_set* neg = isl_aff_neg_basic_set(isl_aff_copy(arg));
            neg = isl_basic_set_intersect(neg, isl_basic_set_copy(context));
            nonneg = isl_basic_set_is_empty(neg) == isl_bool_true;
            isl_basic_set_free(neg);
        }
        if (!nonneg) {
            isl_aff_free(arg);
            return nullptr;
        }
    }
    isl_val* magnitude = isl_val_abs(isl_val_from_integer(ctx, b));
    if (name == "imod") {
        return isl_aff_mod_val(arg, magnitude);
    }
    isl_aff* quotient = isl_aff_floor(isl_aff_scale_down_val(arg, magnitude));
    return b.is_negative() && name == "idiv" ? isl_aff_neg(quotient) : quotient;
}

// Returns null for non-affine products, rationals, powers, unknown functions or names. `context` (optional,
// not consumed) is a domain known to hold; C division of a numerator non-negative there is then the
// quasi-affine floor division instead of a piecewise split on the sign of the numerator. With `piecewise`
// set, only (quasi-)affine expressions are converted: others set it and return null without building pieces.
isl_pw_aff* expression_to_isl_pw_aff(
    const Expression& e,
    isl_local_space* ls,
    const IslNameTable& names,
    isl_basic_set* context = nullptr,
    bool* piecewise = nullptr
) {
    if (isl_aff* aff = expression_to_isl_aff(e, ls, names, context)) {
        return isl_pw_aff_from_aff(aff);
    }
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
    if ((is_max || is_min) && piecewise) {
        *piecewise = true;
        return nullptr;
    }
    if (is_add || is_max || is_min) {
        isl_pw_aff* acc = nullptr;
        for (auto& arg : e->get_args()) {
            isl_pw_aff* term = expression_to_isl_pw_aff(arg, ls, names, context, piecewise);
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
        isl_pw_aff* base = expression_to_isl_pw_aff(dict.begin()->first, ls, names, context, piecewise);
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
            if (piecewise) {
                *piecewise = true;
                return nullptr;
            }
            isl_pw_aff* arg = expression_to_isl_pw_aff(args[0], ls, names, context, piecewise);
            if (!arg) {
                return nullptr;
            }
            // Copy before neg: argument evaluation order is unspecified and neg consumes `arg`.
            isl_pw_aff* neg = isl_pw_aff_neg(isl_pw_aff_copy(arg));
            return isl_pw_aff_max(arg, neg);
        }
        if ((name == "idiv" || name == "imod") && args.size() == 2 && is_nonzero_integer(args[1])) {
            isl_pw_aff* arg = expression_to_isl_pw_aff(args[0], ls, names, context, piecewise);
            if (!arg) {
                return nullptr;
            }
            // C `/` and `%` truncate: a / b == -(a / |b|) and a % b == a % |b| for b < 0.
            const auto& b = SymEngine::down_cast<const SymEngine::Integer&>(*args[1]);
            bool negative = b.is_negative();
            isl_val* magnitude = isl_val_abs(isl_val_from_integer(ctx, b));
            if (nonneg_in(arg, context)) {
                if (name == "imod") {
                    return isl_pw_aff_mod_val(arg, magnitude);
                }
                isl_pw_aff* quotient = isl_pw_aff_floor(isl_pw_aff_scale_down_val(arg, magnitude));
                return negative ? isl_pw_aff_neg(quotient) : quotient;
            }
            if (piecewise) {
                isl_pw_aff_free(arg);
                isl_val_free(magnitude);
                *piecewise = true;
                return nullptr;
            }
            isl_pw_aff* divisor = isl_pw_aff_from_aff(isl_aff_val_on_domain(isl_local_space_copy(ls), magnitude));
            if (name == "imod") {
                return isl_pw_aff_tdiv_r(arg, divisor);
            }
            isl_pw_aff* quotient = isl_pw_aff_tdiv_q(arg, divisor);
            return negative ? isl_pw_aff_neg(quotient) : quotient;
        }
        if (name == floordiv_name && args.size() == 2 && is_nonzero_integer(args[1])) {
            isl_pw_aff* arg = expression_to_isl_pw_aff(args[0], ls, names, context, piecewise);
            if (!arg) {
                return nullptr;
            }
            return isl_pw_aff_floor(isl_pw_aff_scale_down_val(
                arg, isl_val_from_integer(ctx, SymEngine::down_cast<const SymEngine::Integer&>(*args[1]))
            ));
        }
    }
    return nullptr;
}

// Terms whose min (is_min) or max equals `expr`, distributing over sums, integer scalings and idiv by a
// positive constant (all monotone, negative scalings swapping min and max). `x <= min(a, b)` and
// `max(a, b) <= x` are then plain conjunctions instead of piecewise sets whose disjuncts multiply on every
// intersection.
std::vector<Expression> split_extremum(const Expression& expr, bool is_min) {
    constexpr size_t max_pieces = 16;
    if ((is_min && SymEngine::is_a<SymEngine::Min>(*expr)) || (!is_min && SymEngine::is_a<SymEngine::Max>(*expr))) {
        std::vector<Expression> out;
        for (auto& arg : expr->get_args()) {
            auto pieces = split_extremum(arg, is_min);
            out.insert(out.end(), pieces.begin(), pieces.end());
        }
        return out.size() <= max_pieces ? out : std::vector<Expression>{expr};
    }
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        std::vector<Expression> sums = {symbolic::zero()};
        for (auto& term : expr->get_args()) {
            auto pieces = split_extremum(term, is_min);
            if (sums.size() * pieces.size() > max_pieces) {
                return {expr};
            }
            std::vector<Expression> next;
            for (auto& sum : sums) {
                for (auto& piece : pieces) {
                    next.push_back(symbolic::add(sum, piece));
                }
            }
            sums = std::move(next);
        }
        return sums;
    }
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        auto coef = SymEngine::down_cast<const SymEngine::Mul&>(*expr).get_coef();
        auto rest = SymEngine::div(expr, coef);
        if (SymEngine::is_a<SymEngine::Integer>(*coef) && !symbolic::eq(rest, expr)) {
            // A negative scaling swaps min and max.
            bool positive = SymEngine::down_cast<const SymEngine::Integer&>(*coef).is_positive();
            std::vector<Expression> out;
            for (auto& piece : split_extremum(rest, positive ? is_min : !is_min)) {
                out.push_back(symbolic::mul(coef, piece));
            }
            return out;
        }
    }
    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        auto& func = SymEngine::down_cast<const SymEngine::FunctionSymbol&>(*expr);
        auto args = func.get_args();
        if (func.get_name() == "idiv" && args.size() == 2 && SymEngine::is_a<SymEngine::Integer>(*args[1]) &&
            SymEngine::down_cast<const SymEngine::Integer&>(*args[1]).is_positive()) {
            auto pieces = split_extremum(args[0], is_min);
            // A literal piece c >= 0 (max) or c <= 0 (min) dominates every other piece x whose C division
            // differs from floor (max) or ceil (min) division, i.e. x < 0 (max) or x > 0 (min), so those
            // quasi-affine forms are exact, unlike C division which splits on the sign of x.
            bool has_bound = std::any_of(pieces.begin(), pieces.end(), [&](const Expression& p) {
                if (!SymEngine::is_a<SymEngine::Integer>(*p)) {
                    return false;
                }
                auto& c = SymEngine::down_cast<const SymEngine::Integer&>(*p);
                return is_min ? !c.is_positive() : !c.is_negative();
            });
            std::vector<Expression> out;
            for (auto& piece : pieces) {
                if (!has_bound || SymEngine::is_a<SymEngine::Integer>(*piece)) {
                    out.push_back(symbolic::div(piece, args[1]));
                } else if (!is_min) {
                    out.push_back(SymEngine::function_symbol(floordiv_name, {piece, args[1]}));
                } else {
                    auto neg = symbolic::mul(symbolic::integer(-1), piece);
                    out.push_back(
                        symbolic::mul(symbolic::integer(-1), SymEngine::function_symbol(floordiv_name, {neg, args[1]}))
                    );
                }
            }
            return out;
        }
    }
    return {expr};
}

// Builds `{ [dims] -> [exprs] : constraints and strides }` like the corresponding parsed string.
isl_map* build_access_map(
    isl_ctx* ctx,
    const IntersectionSetup& setup,
    const MultiExpression& exprs,
    const ExpressionSet& constraints,
    const Assumptions& assums,
    const std::string& suffix,
    bool* exact = nullptr
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

    // Conjuncts `lhs (<|<=|==|!=) rhs`; those not affine without context are retried against the affine domain.
    enum class Rel { lt, le, eq, ne };
    struct Conjunct {
        Expression lhs, rhs;
        Rel rel;
    };
    std::vector<Conjunct> deferred;
    // Intersects an affine conjunct into bset. A non-affine one returns false unless final, then an (in)equality
    // goes into extra_sets and an inequality is dropped.
    auto add_conjunct = [&](const Conjunct& c, isl_basic_set* context, bool final) {
        // Inequalities still piecewise in the full affine context (e.g. the tight `ub - imod(ub - init, step)`
        // bounds of strided loops, redundant with `ub` and the stride) are dropped: over-approximating the
        // domain is sound for may-dependences and avoids multiplying the disjuncts of every map.
        bool inequality = c.rel == Rel::lt || c.rel == Rel::le;
        bool piecewise = false;
        bool* affine_only = (!final || inequality) ? &piecewise : nullptr;
        isl_pw_aff* lhs = expression_to_isl_pw_aff(c.lhs, ls, names, context, affine_only);
        isl_pw_aff* rhs = lhs ? expression_to_isl_pw_aff(c.rhs, ls, names, context, affine_only) : nullptr;
        if (!lhs || !rhs) {
            isl_pw_aff_free(lhs);
            isl_pw_aff_free(rhs);
            if (piecewise) {
                if (final && exact) {
                    *exact = false;
                }
                return final;
            }
            ok = false;
            return true;
        }
        if (c.rel != Rel::ne && isl_pw_aff_isa_aff(lhs) == isl_bool_true && isl_pw_aff_isa_aff(rhs) == isl_bool_true) {
            isl_aff* l = isl_pw_aff_as_aff(lhs);
            isl_aff* r = isl_pw_aff_as_aff(rhs);
            isl_basic_set* s = c.rel == Rel::eq   ? isl_aff_eq_basic_set(l, r)
                               : c.rel == Rel::lt ? isl_aff_lt_basic_set(l, r)
                                                  : isl_aff_le_basic_set(l, r);
            bset = isl_basic_set_intersect(bset, s);
            return true;
        }
        if (!final) {
            isl_pw_aff_free(lhs);
            isl_pw_aff_free(rhs);
            return false;
        }
        isl_set* s = c.rel == Rel::ne ? isl_pw_aff_ne_set(lhs, rhs) : isl_pw_aff_eq_set(lhs, rhs);
        extra_sets = extra_sets ? isl_set_intersect(extra_sets, s) : s;
        return true;
    };

    for (auto it = constraints.begin(); ok && it != constraints.end(); ++it) {
        auto& con = *it;
        bool is_lt = SymEngine::is_a<SymEngine::StrictLessThan>(*con);
        bool is_le = SymEngine::is_a<SymEngine::LessThan>(*con);
        bool is_ne = SymEngine::is_a<SymEngine::Unequality>(*con);
        bool is_eq = SymEngine::is_a<SymEngine::Equality>(*con);
        if (!is_lt && !is_le && !is_ne && !is_eq) {
            continue;
        }
        Rel rel = is_lt ? Rel::lt : is_le ? Rel::le : is_eq ? Rel::eq : Rel::ne;
        auto args = con->get_args();
        if (SymEngine::is_a<SymEngine::Infty>(*args[0]) || SymEngine::is_a<SymEngine::Infty>(*args[1])) {
            continue;
        }
        // lhs <= rhs holds iff every max-piece of lhs is <= every min-piece of rhs.
        std::vector<Expression> lhs_pieces = {args[0]};
        std::vector<Expression> rhs_pieces = {args[1]};
        if (is_lt || is_le) {
            lhs_pieces = split_extremum(args[0], /*is_min=*/false);
            rhs_pieces = split_extremum(args[1], /*is_min=*/true);
        }
        for (auto& lhs_expr : lhs_pieces) {
            for (auto& rhs_expr : rhs_pieces) {
                if (!ok) {
                    break;
                }
                Conjunct c{lhs_expr, rhs_expr, rel};
                if (!add_conjunct(c, nullptr, false)) {
                    deferred.push_back(std::move(c));
                }
            }
        }
    }

    // `exists it : dim = lb + it * step`, emitted for constant non-unit strides.
    for (size_t i = 0; ok && i < n_dims; i++) {
        auto sym = symbolic::symbol(setup.dimensions[i]);
        auto assum = assums.find(sym);
        auto map_func = assum == assums.end() ? Expression(SymEngine::null) : assum->second.map();
        if (map_func.is_null() || symbolic::eq(map_func, symbolic::add(sym, symbolic::one()))) {
            continue;
        }
        // Other updates leave the stride out of the domain.
        bool exact_before = exact && *exact;
        if (exact) {
            *exact = false;
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
        if (!symbolic::eq(arg0, sym)) {
            arg0 = args[1];
            arg1 = args[0];
        }
        if (!symbolic::eq(arg0, sym) || !SymEngine::is_a<SymEngine::Integer>(*arg1) ||
            symbolic::eq(arg1, symbolic::one())) {
            continue;
        }
        auto lb = stride_residue(assums.at(sym).tight_lower_bound(), arg1);
        if (lb.is_null()) {
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
        if (exact) {
            *exact = exact_before;
        }
    }

    // Each conjunct that becomes affine strengthens the context of the remaining ones.
    for (bool progress = true; ok && progress;) {
        progress = false;
        for (size_t k = 0; ok && k < deferred.size();) {
            if (add_conjunct(deferred[k], bset, false)) {
                deferred.erase(deferred.begin() + k);
                progress = true;
            } else {
                k++;
            }
        }
    }
    for (size_t k = 0; ok && k < deferred.size(); k++) {
        add_conjunct(deferred[k], bset, true);
    }

    for (size_t k = 0; ok && k < exprs.size(); k++) {
        isl_pw_aff* pa = expression_to_isl_pw_aff(exprs[k], ls, names, bset);
        if (!pa) {
            ok = false;
            break;
        }
        outs_affine = outs_affine && isl_pw_aff_isa_aff(pa) == isl_bool_true;
        outs = isl_pw_aff_list_add(outs, pa);
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

// Dimensions and parameters as in expression_to_map_str; `coupled` adds the constraints registered on the symbols.
IntersectionSetup single_setup(const MultiExpression& expr, const Assumptions& assums, bool coupled) {
    IntersectionSetup setup;
    SymbolSet syms;
    for (auto& e : expr) {
        for (auto& s : symbolic::atoms(e)) {
            syms.insert(s);
        }
    }
    SymbolSet params;
    for (auto& sym : syms) {
        auto it = assums.find(sym);
        if (it != assums.end() && it->second.constant() && it->second.map().is_null()) {
            params.insert(sym);
        } else {
            setup.dimensions.push_back(sym->get_name());
            setup.dimensions_syms.insert(sym);
        }
    }
    SymbolSet seen;
    setup.constraints_syms_1 = generate_constraints(syms, assums, seen);
    if (coupled) {
        // Close over the symbols the registered constraints mention.
        SymbolSet done;
        for (bool grew = true; grew;) {
            grew = false;
            SymbolSet pending;
            for (auto& sym : seen) {
                if (!done.insert(sym).second) {
                    continue;
                }
                for (auto& c : assums.at(sym).constraints()) {
                    setup.constraints_syms_1.insert(symbolic::Le(c, symbolic::zero()));
                    for (auto& s : symbolic::atoms(c)) {
                        if (!seen.count(s) && assums.count(s)) {
                            pending.insert(s);
                        }
                    }
                }
            }
            if (!pending.empty()) {
                auto more = generate_constraints(pending, assums, seen);
                setup.constraints_syms_1.insert(more.begin(), more.end());
                grew = true;
            }
        }
    }
    for (auto& con : setup.constraints_syms_1) {
        for (auto& s : symbolic::atoms(con)) {
            if (!setup.dimensions_syms.count(s)) {
                params.insert(s);
            }
        }
    }
    for (auto& p : params) {
        setup.parameters.push_back(p->get_name());
    }
    std::sort(setup.parameters.begin(), setup.parameters.end());
    return setup;
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

isl_map* expression_to_may_map(isl_ctx* ctx, const MultiExpression& expr, const Assumptions& assums) {
    auto setup = single_setup(expr, assums, /*coupled=*/false);
    return build_access_map(ctx, setup, expr, setup.constraints_syms_1, assums, "");
}

namespace {

// Folds `e` with integer symbol values to an integer, or null.
Expression fold_at(const Expression& e, const std::unordered_map<std::string, Expression>& values) {
    if (SymEngine::is_a<SymEngine::Integer>(*e)) {
        return e;
    }
    if (SymEngine::is_a<SymEngine::Symbol>(*e)) {
        auto it = values.find(SymEngine::down_cast<const SymEngine::Symbol&>(*e).get_name());
        return it == values.end() ? Expression(SymEngine::null) : it->second;
    }
    SymEngine::vec_basic args;
    for (auto& arg : e->get_args()) {
        auto folded = fold_at(arg, values);
        if (folded.is_null()) {
            return SymEngine::null;
        }
        args.push_back(folded);
    }
    Expression result = SymEngine::null;
    if (SymEngine::is_a<SymEngine::Add>(*e)) {
        result = SymEngine::add(args);
    } else if (SymEngine::is_a<SymEngine::Mul>(*e)) {
        result = SymEngine::mul(args);
    } else if (SymEngine::is_a<SymEngine::Min>(*e)) {
        result = SymEngine::min(args);
    } else if (SymEngine::is_a<SymEngine::Max>(*e)) {
        result = SymEngine::max(args);
    } else if (SymEngine::is_a<SymEngine::Pow>(*e)) {
        result = SymEngine::pow(args[0], args[1]);
    } else if (SymEngine::is_a<SymEngine::FunctionSymbol>(*e) && args.size() == 2) {
        // Arbitrary precision: sample points may lie near the int64 limits of the assumptions.
        auto& name = SymEngine::down_cast<const SymEngine::FunctionSymbol&>(*e).get_name();
        if ((name == "idiv" || name == "imod") && !symbolic::eq(args[1], symbolic::zero())) {
            auto& a = SymEngine::down_cast<const SymEngine::Integer&>(*args[0]);
            auto& b = SymEngine::down_cast<const SymEngine::Integer&>(*args[1]);
            auto q = SymEngine::quotient(a, b);
            result = name == "idiv" ? Expression(q) : symbolic::sub(args[0], symbolic::mul(args[1], q));
        }
    } else if (SymEngine::is_a<SymEngine::FunctionSymbol>(*e) && args.size() == 1 &&
               SymEngine::down_cast<const SymEngine::FunctionSymbol&>(*e).get_name() == "iabs") {
        result = SymEngine::abs(args[0]);
    }
    return !result.is_null() && SymEngine::is_a<SymEngine::Integer>(*result) ? result : Expression(SymEngine::null);
}

// Whether the point satisfies every constraint and stride of the setup; false if undecidable.
bool satisfies(
    const IntersectionSetup& setup, const Assumptions& assums, const std::unordered_map<std::string, Expression>& values
) {
    for (auto& con : setup.constraints_syms_1) {
        auto args = con->get_args();
        if (args.size() != 2) {
            continue;
        }
        if (SymEngine::is_a<SymEngine::Infty>(*args[0]) || SymEngine::is_a<SymEngine::Infty>(*args[1])) {
            continue;
        }
        auto lhs = fold_at(args[0], values);
        auto rhs = fold_at(args[1], values);
        if (lhs.is_null() || rhs.is_null()) {
            return false;
        }
        bool holds;
        if (SymEngine::is_a<SymEngine::StrictLessThan>(*con)) {
            holds = symbolic::is_true(symbolic::Lt(lhs, rhs));
        } else if (SymEngine::is_a<SymEngine::LessThan>(*con)) {
            holds = symbolic::is_true(symbolic::Le(lhs, rhs));
        } else if (SymEngine::is_a<SymEngine::Equality>(*con)) {
            holds = symbolic::eq(lhs, rhs);
        } else if (SymEngine::is_a<SymEngine::Unequality>(*con)) {
            holds = !symbolic::eq(lhs, rhs);
        } else {
            continue;
        }
        if (!holds) {
            return false;
        }
    }
    for (auto& dim : setup.dimensions) {
        auto sym = symbolic::symbol(dim);
        auto it = assums.find(sym);
        if (it == assums.end() || it->second.map().is_null()) {
            continue;
        }
        auto step = symbolic::sub(it->second.map(), sym);
        auto init = it->second.tight_lower_bound();
        if (!SymEngine::is_a<SymEngine::Integer>(*step)) {
            return false;
        }
        if (symbolic::eq(step, symbolic::one())) {
            continue;
        }
        auto lb = init.is_null() ? init : fold_at(init, values);
        if (lb.is_null() || !symbolic::eq(symbolic::mod(symbolic::sub(values.at(dim), lb), step), symbolic::zero())) {
            return false;
        }
    }
    return true;
}

} // namespace

std::optional<bool> decide_nonneg(const Expression& expr, const Assumptions& assums, bool strict) {
    auto setup = single_setup({expr}, assums, /*coupled=*/true);
    polyhedral::IslCtx ctx;
    bool exact = true;
    isl_map* map = ctx ? build_access_map(ctx.get(), setup, {expr}, setup.constraints_syms_1, assums, "", &exact)
                       : nullptr;
    if (!map) {
        return std::nullopt;
    }
    // Iterations whose value violates the goal.
    isl_set* bad =
        isl_set_upper_bound_si(isl_set_universe(isl_space_range(isl_map_get_space(map))), isl_dim_set, 0, strict ? 0 : -1);
    isl_set* violations = isl_map_domain(isl_map_intersect_range(map, bad));
    isl_bool empty = isl_set_is_empty(violations);
    std::optional<bool> result;
    if (empty == isl_bool_true) {
        result = true;
    } else if (empty == isl_bool_false && exact) {
        result = false;
    } else if (empty == isl_bool_false) {
        // The domain over-approximates; a violation satisfying all assumptions still refutes the goal.
        isl_point* point = isl_set_sample_point(isl_set_copy(violations));
        std::unordered_map<std::string, Expression> values;
        bool ok = point != nullptr;
        for (auto [type, names] :
             {std::make_pair(isl_dim_param, &setup.parameters), std::make_pair(isl_dim_set, &setup.dimensions)}) {
            for (size_t i = 0; ok && i < names->size(); i++) {
                int pos = type == isl_dim_param
                              ? isl_set_find_dim_by_name(violations, isl_dim_param, (*names)[i].c_str())
                              : int(i);
                isl_val* v = pos >= 0 ? isl_point_get_coordinate_val(point, type, pos) : nullptr;
                char* str = v && isl_val_is_int(v) ? isl_val_to_str(v) : nullptr;
                ok = str != nullptr;
                if (ok) {
                    try {
                        values.emplace((*names)[i], symbolic::integer(std::stoll(str)));
                    } catch (const std::out_of_range&) {
                        ok = false;
                    }
                }
                free(str);
                isl_val_free(v);
            }
        }
        isl_point_free(point);
        if (ok && satisfies(setup, assums, values)) {
            result = false;
        }
    }
    isl_set_free(violations);
    return result;
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
