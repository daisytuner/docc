#include "sdfg/symbolic/utils.h"

#include <isl/ctx.h>
#include <isl/set.h>
#include <isl/space.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

// ISL-compatible printer that converts idiv to floord
class ISLSymbolicPrinter : public SymEngine::BaseVisitor<ISLSymbolicPrinter, SymEngine::CodePrinter> {
public:
    using SymEngine::CodePrinter::apply;
    using SymEngine::CodePrinter::bvisit;
    using SymEngine::CodePrinter::str_;

    void bvisit(const SymEngine::FunctionSymbol& x) {
        if (x.get_name() == "idiv") {
            // ISL uses floord(a, b) for floor division
            str_ = "floord(" + apply(x.get_args()[0]) + ", " + apply(x.get_args()[1]) + ")";
        } else if (x.get_name() == "iabs") {
            // ISL doesn't support abs directly, but we can express it
            str_ = apply(x.get_args()[0]); // Simplify: assume non-negative for ISL constraints
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

std::tuple<std::string, std::string, std::string> expressions_to_intersection_map_str(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
) {
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
    std::vector<std::string> dimensions;
    SymbolSet dimensions_syms;
    std::vector<std::string> parameters;
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
    auto constraints_syms_1 = generate_constraints(syms, assums1, seen);
    seen.clear();
    auto constraints_syms_2 = generate_constraints(syms, assums2, seen);

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

} // namespace symbolic
} // namespace sdfg
