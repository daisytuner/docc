#include "sdfg/transformations/loop_condition_normalize.h"

#include <stdexcept>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"

/**
 * Loop Condition Normalize Transformation Implementation
 *
 * This transformation normalizes loop conditions through multiple steps:
 *
 * 1. Pre-normalization: Simplify boolean comparisons
 *    - `false == (relational)` → negated relational
 *    - `true == (relational)` → relational
 *
 * 2. Indvar isolation: Ensure bare indvar on LHS of all relationals
 *    - `i + offset < bound` → `i < bound - offset`
 *    - `bound < i + offset` → `bound - offset < i`
 *    (No stride requirement - pure algebraic transformation)
 *
 * 3. Main normalization: Convert `!=` to `<` or `>` based on stride
 *    - stride = +1: `i != N` → `i < N`
 *    - stride = -1: `i != N` → `i > N`
 *    (Requires unit stride ±1)
 *
 * 4. Post-normalization: Simplify trivial max/min in bounds
 *    - `i < max(init, N)` with loop init=init → `i < N`
 *
 * The transformation assumes LLVM-style well-formed loops where the bound is
 * reachable from init.
 */

namespace sdfg {
namespace transformations {

namespace {

/**
 * Negate a relational expression
 * LessThan(a, b) → Ge(a, b)
 * StrictLessThan(a, b) → Le(a, b) (actually Gt reversed: b > a means a <= b? No...)
 * Actually: Not(a < b) means a >= b
 */
symbolic::Condition negate_relational(const symbolic::Condition& rel) {
    if (SymEngine::is_a<SymEngine::LessThan>(*rel)) {
        auto lt = SymEngine::rcp_static_cast<const SymEngine::LessThan>(rel);
        return symbolic::Gt(lt->get_arg1(), lt->get_arg2());
    }
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*rel)) {
        auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(rel);
        return symbolic::Ge(lt->get_arg1(), lt->get_arg2());
    }
    // Fallback: wrap in Not
    return symbolic::Not(rel);
}

/**
 * Pre-normalization: Simplify `false == (relational)` and `true == (relational)`
 * Returns the simplified condition.
 */
symbolic::Condition simplify_boolean_comparisons(const symbolic::Condition& cond) {
    // Handle Equality: (true == rel) → rel, (false == rel) → Not(rel)
    if (SymEngine::is_a<SymEngine::Equality>(*cond)) {
        auto eq = SymEngine::rcp_static_cast<const SymEngine::Equality>(cond);
        auto arg1 = eq->get_arg1();
        auto arg2 = eq->get_arg2();

        // Check if one side is true/false and the other is relational
        if (SymEngine::is_a_Relational(*arg1) && !SymEngine::is_a_Relational(*arg2)) {
            if (symbolic::is_true(arg2)) {
                return SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg1);
            } else if (symbolic::is_false(arg2)) {
                return negate_relational(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg1));
            }
        }
        if (SymEngine::is_a_Relational(*arg2) && !SymEngine::is_a_Relational(*arg1)) {
            if (symbolic::is_true(arg1)) {
                return SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg2);
            } else if (symbolic::is_false(arg1)) {
                return negate_relational(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg2));
            }
        }
    }

    // Handle Unequality: (true != rel) → Not(rel), (false != rel) → rel
    if (SymEngine::is_a<SymEngine::Unequality>(*cond)) {
        auto ne = SymEngine::rcp_static_cast<const SymEngine::Unequality>(cond);
        auto arg1 = ne->get_arg1();
        auto arg2 = ne->get_arg2();

        if (SymEngine::is_a_Relational(*arg1) && !SymEngine::is_a_Relational(*arg2)) {
            if (symbolic::is_true(arg2)) {
                return negate_relational(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg1));
            } else if (symbolic::is_false(arg2)) {
                return SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg1);
            }
        }
        if (SymEngine::is_a_Relational(*arg2) && !SymEngine::is_a_Relational(*arg1)) {
            if (symbolic::is_true(arg1)) {
                return negate_relational(SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg2));
            } else if (symbolic::is_false(arg1)) {
                return SymEngine::rcp_static_cast<const SymEngine::Boolean>(arg2);
            }
        }
    }

    // Handle And/Or recursively
    if (SymEngine::is_a<SymEngine::And>(*cond)) {
        auto and_cond = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
        auto args = and_cond->get_args();
        std::vector<symbolic::Condition> new_args;
        for (const auto& arg : args) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            new_args.push_back(simplify_boolean_comparisons(bool_arg));
        }
        symbolic::Condition result = new_args[0];
        for (size_t i = 1; i < new_args.size(); ++i) {
            result = symbolic::And(result, new_args[i]);
        }
        return result;
    }

    if (SymEngine::is_a<SymEngine::Or>(*cond)) {
        auto or_cond = SymEngine::rcp_static_cast<const SymEngine::Or>(cond);
        auto args = or_cond->get_args();
        std::vector<symbolic::Condition> new_args;
        for (const auto& arg : args) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            new_args.push_back(simplify_boolean_comparisons(bool_arg));
        }
        symbolic::Condition result = new_args[0];
        for (size_t i = 1; i < new_args.size(); ++i) {
            result = symbolic::Or(result, new_args[i]);
        }
        return result;
    }

    return cond;
}

/**
 * Simplify max(init, bound) → bound when one argument equals loop init.
 * This is safe because if init <= bound, the loop will terminate properly.
 */
symbolic::Expression simplify_max_with_init(const symbolic::Expression& expr, const symbolic::Expression& loop_init) {
    if (SymEngine::is_a<SymEngine::Max>(*expr)) {
        auto max_expr = SymEngine::rcp_static_cast<const SymEngine::Max>(expr);
        auto args = max_expr->get_args();

        // Find if any argument equals loop_init
        for (const auto& arg : args) {
            if (symbolic::eq(arg, loop_init)) {
                // Collect remaining arguments
                std::vector<symbolic::Expression> remaining;
                for (const auto& other : args) {
                    if (!symbolic::eq(other, loop_init)) {
                        remaining.push_back(other);
                    }
                }
                if (remaining.size() == 1) {
                    return remaining[0];
                } else if (remaining.size() > 1) {
                    return SymEngine::max(remaining);
                }
            }
        }
    }
    return expr;
}

/**
 * Post-normalization: Simplify bounds containing max(init, ...) patterns
 */
symbolic::Condition simplify_max_bounds(const symbolic::Condition& cond, const symbolic::Expression& loop_init) {
    // Handle StrictLessThan: i < max(init, N) → i < N
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*cond)) {
        auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(cond);
        auto lhs = lt->get_arg1();
        auto rhs = lt->get_arg2();
        auto simplified_rhs = simplify_max_with_init(rhs, loop_init);
        if (!symbolic::eq(rhs, simplified_rhs)) {
            return symbolic::Lt(lhs, simplified_rhs);
        }
    }

    // Handle LessThan: i <= max(init, N) → i <= N
    if (SymEngine::is_a<SymEngine::LessThan>(*cond)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(cond);
        auto lhs = le->get_arg1();
        auto rhs = le->get_arg2();
        auto simplified_rhs = simplify_max_with_init(rhs, loop_init);
        if (!symbolic::eq(rhs, simplified_rhs)) {
            return symbolic::Le(lhs, simplified_rhs);
        }
    }

    // Handle And/Or recursively
    if (SymEngine::is_a<SymEngine::And>(*cond)) {
        auto and_cond = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
        auto args = and_cond->get_args();
        std::vector<symbolic::Condition> new_args;
        for (const auto& arg : args) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            new_args.push_back(simplify_max_bounds(bool_arg, loop_init));
        }
        symbolic::Condition result = new_args[0];
        for (size_t i = 1; i < new_args.size(); ++i) {
            result = symbolic::And(result, new_args[i]);
        }
        return result;
    }

    if (SymEngine::is_a<SymEngine::Or>(*cond)) {
        auto or_cond = SymEngine::rcp_static_cast<const SymEngine::Or>(cond);
        auto args = or_cond->get_args();
        std::vector<symbolic::Condition> new_args;
        for (const auto& arg : args) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            new_args.push_back(simplify_max_bounds(bool_arg, loop_init));
        }
        symbolic::Condition result = new_args[0];
        for (size_t i = 1; i < new_args.size(); ++i) {
            result = symbolic::Or(result, new_args[i]);
        }
        return result;
    }

    return cond;
}

/**
 * Check if condition contains boolean comparison patterns (false == rel, true == rel)
 */
bool has_boolean_comparison_pattern(const symbolic::Condition& cond) {
    if (SymEngine::is_a<SymEngine::Equality>(*cond) || SymEngine::is_a<SymEngine::Unequality>(*cond)) {
        auto rel = SymEngine::rcp_static_cast<const SymEngine::Relational>(cond);
        auto arg1 = rel->get_arg1();
        auto arg2 = rel->get_arg2();

        bool arg1_is_bool = symbolic::is_true(arg1) || symbolic::is_false(arg1);
        bool arg2_is_bool = symbolic::is_true(arg2) || symbolic::is_false(arg2);
        bool arg1_is_rel = SymEngine::is_a_Relational(*arg1);
        bool arg2_is_rel = SymEngine::is_a_Relational(*arg2);

        if ((arg1_is_bool && arg2_is_rel) || (arg2_is_bool && arg1_is_rel)) {
            return true;
        }
    }

    // Check recursively in And/Or
    if (SymEngine::is_a<SymEngine::And>(*cond)) {
        auto and_cond = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
        for (const auto& arg : and_cond->get_args()) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            if (has_boolean_comparison_pattern(bool_arg)) {
                return true;
            }
        }
    }
    if (SymEngine::is_a<SymEngine::Or>(*cond)) {
        auto or_cond = SymEngine::rcp_static_cast<const SymEngine::Or>(cond);
        for (const auto& arg : or_cond->get_args()) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            if (has_boolean_comparison_pattern(bool_arg)) {
                return true;
            }
        }
    }

    return false;
}

/**
 * Try to isolate indvar in a relational expression.
 * Transforms: (indvar + offset) <relop> bound → indvar <relop> (bound - offset)
 *             bound <relop> (indvar + offset) → (bound - offset) <relop> indvar
 * Returns the original condition if isolation is not applicable.
 */
symbolic::Condition isolate_indvar_in_relational(const symbolic::Condition& cond, const symbolic::Symbol& indvar) {
    // Helper to extract offset from affine expression with indvar coefficient = 1
    auto extract_offset = [&indvar](const symbolic::Expression& expr) -> std::optional<symbolic::Expression> {
        symbolic::SymbolVec syms = {indvar};
        auto poly = symbolic::polynomial(expr, syms);
        if (poly.is_null()) {
            return std::nullopt;
        }

        auto coeffs = symbolic::affine_coefficients(poly);
        if (coeffs.empty() || coeffs.find(indvar) == coeffs.end()) {
            return std::nullopt;
        }

        auto coeff = coeffs.at(indvar);
        if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
            return std::nullopt;
        }
        auto coeff_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(coeff)->as_int();
        if (coeff_int != 1) {
            return std::nullopt;
        }

        symbolic::Expression offset = symbolic::zero();
        if (coeffs.count(symbolic::symbol("__daisy_constant__"))) {
            offset = coeffs.at(symbolic::symbol("__daisy_constant__"));
        }
        return offset;
    };

    // Helper to process a binary relational
    auto process_relational =
        [&](const symbolic::Expression& lhs, const symbolic::Expression& rhs, auto make_same_rel, auto make_flipped_rel
        ) -> symbolic::Condition {
        bool lhs_has_indvar = symbolic::uses(lhs, indvar->get_name());
        bool rhs_has_indvar = symbolic::uses(rhs, indvar->get_name());

        // Skip if indvar not present or on both sides
        if ((!lhs_has_indvar && !rhs_has_indvar) || (lhs_has_indvar && rhs_has_indvar)) {
            return cond;
        }

        if (lhs_has_indvar) {
            // (indvar + offset) <relop> rhs → indvar <relop> (rhs - offset)
            if (auto offset = extract_offset(lhs)) {
                if (!symbolic::eq(*offset, symbolic::zero())) {
                    auto new_bound = symbolic::expand(symbolic::sub(rhs, *offset));
                    return make_same_rel(indvar, new_bound);
                }
            }
        } else {
            // lhs <relop> (indvar + offset) → (lhs - offset) <relop> indvar
            if (auto offset = extract_offset(rhs)) {
                if (!symbolic::eq(*offset, symbolic::zero())) {
                    auto new_bound = symbolic::expand(symbolic::sub(lhs, *offset));
                    return make_flipped_rel(new_bound, indvar);
                }
            }
        }
        return cond;
    };

    // Handle StrictLessThan: a < b
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*cond)) {
        auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(cond);
        return process_relational(
            lt->get_arg1(),
            lt->get_arg2(),
            [](auto a, auto b) { return symbolic::Lt(a, b); },
            [](auto a, auto b) { return symbolic::Lt(a, b); }
        );
    }

    // Handle LessThan (<=): a <= b
    if (SymEngine::is_a<SymEngine::LessThan>(*cond)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(cond);
        return process_relational(
            le->get_arg1(),
            le->get_arg2(),
            [](auto a, auto b) { return symbolic::Le(a, b); },
            [](auto a, auto b) { return symbolic::Le(a, b); }
        );
    }

    // Handle Unequality: a != b
    if (SymEngine::is_a<SymEngine::Unequality>(*cond)) {
        auto ne = SymEngine::rcp_static_cast<const SymEngine::Unequality>(cond);
        return process_relational(
            ne->get_arg1(),
            ne->get_arg2(),
            [](auto a, auto b) { return symbolic::Ne(a, b); },
            [](auto a, auto b) { return symbolic::Ne(a, b); }
        );
    }

    // Note: > and >= are typically represented as < and <= with swapped args in SymEngine
    // but handle them for completeness

    return cond;
}

/**
 * Recursively isolate indvar in all relationals within a condition
 */
symbolic::Condition isolate_indvar_in_condition(const symbolic::Condition& cond, const symbolic::Symbol& indvar) {
    // Try isolation on this condition first
    auto isolated = isolate_indvar_in_relational(cond, indvar);
    if (!symbolic::eq(isolated, cond)) {
        return isolated;
    }

    // Handle And recursively
    if (SymEngine::is_a<SymEngine::And>(*cond)) {
        auto and_cond = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
        auto args = and_cond->get_args();
        std::vector<symbolic::Condition> new_args;
        for (const auto& arg : args) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            new_args.push_back(isolate_indvar_in_condition(bool_arg, indvar));
        }
        symbolic::Condition result = new_args[0];
        for (size_t i = 1; i < new_args.size(); ++i) {
            result = symbolic::And(result, new_args[i]);
        }
        return result;
    }

    // Handle Or recursively
    if (SymEngine::is_a<SymEngine::Or>(*cond)) {
        auto or_cond = SymEngine::rcp_static_cast<const SymEngine::Or>(cond);
        auto args = or_cond->get_args();
        std::vector<symbolic::Condition> new_args;
        for (const auto& arg : args) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            new_args.push_back(isolate_indvar_in_condition(bool_arg, indvar));
        }
        symbolic::Condition result = new_args[0];
        for (size_t i = 1; i < new_args.size(); ++i) {
            result = symbolic::Or(result, new_args[i]);
        }
        return result;
    }

    return cond;
}

/**
 * Check if a relational has non-isolated indvar (i + offset <relop> bound)
 */
bool has_non_isolated_indvar(const symbolic::Condition& cond, const symbolic::Symbol& indvar) {
    auto check_expr = [&indvar](const symbolic::Expression& expr) -> bool {
        if (!symbolic::uses(expr, indvar->get_name())) {
            return false;
        }
        // If it uses indvar but isn't just the indvar, it needs isolation
        if (!symbolic::eq(expr, indvar)) {
            // Verify it's affine with coeff=1 (otherwise we can't isolate)
            symbolic::SymbolVec syms = {indvar};
            auto poly = symbolic::polynomial(expr, syms);
            if (poly.is_null()) {
                return false;
            }
            auto coeffs = symbolic::affine_coefficients(poly);
            if (coeffs.find(indvar) == coeffs.end()) {
                return false;
            }
            auto coeff = coeffs.at(indvar);
            if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
                return false;
            }
            auto coeff_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(coeff)->as_int();
            if (coeff_int != 1) {
                return false;
            }
            // Has non-zero offset
            if (coeffs.count(symbolic::symbol("__daisy_constant__"))) {
                auto offset = coeffs.at(symbolic::symbol("__daisy_constant__"));
                if (!symbolic::eq(offset, symbolic::zero())) {
                    return true;
                }
            }
        }
        return false;
    };

    // Check relationals
    if (SymEngine::is_a<SymEngine::StrictLessThan>(*cond)) {
        auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(cond);
        return check_expr(lt->get_arg1()) || check_expr(lt->get_arg2());
    }
    if (SymEngine::is_a<SymEngine::LessThan>(*cond)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(cond);
        return check_expr(le->get_arg1()) || check_expr(le->get_arg2());
    }
    if (SymEngine::is_a<SymEngine::Unequality>(*cond)) {
        auto ne = SymEngine::rcp_static_cast<const SymEngine::Unequality>(cond);
        return check_expr(ne->get_arg1()) || check_expr(ne->get_arg2());
    }

    // Check recursively in And/Or
    if (SymEngine::is_a<SymEngine::And>(*cond)) {
        auto and_cond = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
        for (const auto& arg : and_cond->get_args()) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            if (has_non_isolated_indvar(bool_arg, indvar)) {
                return true;
            }
        }
    }
    if (SymEngine::is_a<SymEngine::Or>(*cond)) {
        auto or_cond = SymEngine::rcp_static_cast<const SymEngine::Or>(cond);
        for (const auto& arg : or_cond->get_args()) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            if (has_non_isolated_indvar(bool_arg, indvar)) {
                return true;
            }
        }
    }

    return false;
}

/**
 * Check if condition contains max(init, bound) patterns in bounds
 */
bool has_max_init_pattern(const symbolic::Condition& cond, const symbolic::Expression& loop_init) {
    auto check_expr = [&loop_init](const symbolic::Expression& expr) -> bool {
        if (SymEngine::is_a<SymEngine::Max>(*expr)) {
            auto max_expr = SymEngine::rcp_static_cast<const SymEngine::Max>(expr);
            for (const auto& arg : max_expr->get_args()) {
                if (symbolic::eq(arg, loop_init)) {
                    return true;
                }
            }
        }
        return false;
    };

    if (SymEngine::is_a<SymEngine::StrictLessThan>(*cond)) {
        auto lt = SymEngine::rcp_static_cast<const SymEngine::StrictLessThan>(cond);
        if (check_expr(lt->get_arg2())) {
            return true;
        }
    }
    if (SymEngine::is_a<SymEngine::LessThan>(*cond)) {
        auto le = SymEngine::rcp_static_cast<const SymEngine::LessThan>(cond);
        if (check_expr(le->get_arg2())) {
            return true;
        }
    }

    // Check recursively in And/Or
    if (SymEngine::is_a<SymEngine::And>(*cond)) {
        auto and_cond = SymEngine::rcp_static_cast<const SymEngine::And>(cond);
        for (const auto& arg : and_cond->get_args()) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            if (has_max_init_pattern(bool_arg, loop_init)) {
                return true;
            }
        }
    }
    if (SymEngine::is_a<SymEngine::Or>(*cond)) {
        auto or_cond = SymEngine::rcp_static_cast<const SymEngine::Or>(cond);
        for (const auto& arg : or_cond->get_args()) {
            auto bool_arg = SymEngine::rcp_dynamic_cast<const SymEngine::Boolean>(arg);
            if (has_max_init_pattern(bool_arg, loop_init)) {
                return true;
            }
        }
    }

    return false;
}

} // anonymous namespace

LoopConditionNormalize::LoopConditionNormalize(structured_control_flow::StructuredLoop& loop) : loop_(loop) {}

std::string LoopConditionNormalize::name() const { return "LoopConditionNormalize"; }

bool LoopConditionNormalize::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Check for unit stride (±1) - required for != conversion
    auto stride = loop_.stride();
    bool has_unit_stride = false;
    if (!stride.is_null()) {
        auto stride_int = stride->as_int();
        has_unit_stride = (stride_int == 1 || stride_int == -1);
    }

    auto condition = loop_.condition();
    auto init = loop_.init();

    // Check for pattern 1: boolean comparison (false == relational, true == relational)
    bool has_boolean_pattern = has_boolean_comparison_pattern(condition);

    // Check for pattern 2: max(init, bound) in bounds
    bool has_max_pattern = has_max_init_pattern(condition, init);

    // Check for pattern 3: non-isolated indvar (i + offset <relop> bound)
    bool has_isolation_pattern = has_non_isolated_indvar(condition, loop_.indvar());

    // Check for pattern 4: != involving indvar (requires unit stride)
    bool has_unequality_pattern = false;
    if (has_unit_stride) {
        symbolic::CNF cnf;
        try {
            cnf = symbolic::conjunctive_normal_form(condition);
        } catch (...) {
            // CNF conversion failed, still check other patterns
            cnf = {};
        }

        auto indvar = loop_.indvar();

        for (const auto& clause : cnf) {
            for (const auto& literal : clause) {
                if (SymEngine::is_a<SymEngine::Unequality>(*literal)) {
                    auto ne = SymEngine::rcp_static_cast<const SymEngine::Unequality>(literal);
                    auto lhs = ne->get_arg1();
                    auto rhs = ne->get_arg2();

                    bool lhs_has_indvar = symbolic::uses(lhs, indvar->get_name());
                    bool rhs_has_indvar = symbolic::uses(rhs, indvar->get_name());

                    if (!lhs_has_indvar && !rhs_has_indvar) {
                        continue;
                    }
                    if (lhs_has_indvar && rhs_has_indvar) {
                        continue;
                    }

                    // Check if affine with coefficient = 1
                    auto expr_with_indvar = lhs_has_indvar ? lhs : rhs;
                    symbolic::SymbolVec syms = {indvar};
                    auto poly = symbolic::polynomial(expr_with_indvar, syms);
                    if (poly.is_null()) {
                        continue;
                    }

                    auto coeffs = symbolic::affine_coefficients(poly);
                    if (coeffs.empty() || coeffs.find(indvar) == coeffs.end()) {
                        continue;
                    }

                    auto coeff = coeffs.at(indvar);
                    if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
                        continue;
                    }
                    auto coeff_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(coeff)->as_int();
                    if (coeff_int != 1) {
                        continue;
                    }

                    has_unequality_pattern = true;
                    break;
                }
            }
            if (has_unequality_pattern) {
                break;
            }
        }
    }

    return has_boolean_pattern || has_max_pattern || has_isolation_pattern || has_unequality_pattern;
}

void LoopConditionNormalize::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto indvar = loop_.indvar();
    auto stride = loop_.stride();
    auto init = loop_.init();

    // Step 1: Pre-normalization - simplify boolean comparisons
    // e.g., (false == (a < b)) → (a >= b)
    symbolic::Condition condition = simplify_boolean_comparisons(loop_.condition());

    // Step 2: Isolate indvar in all relationals (no stride requirement)
    // e.g., (i + offset < bound) → (i < bound - offset)
    condition = isolate_indvar_in_condition(condition, indvar);

    // Step 3: Main normalization - convert != to < or > based on stride
    // Only if stride is unit (±1)
    if (!stride.is_null()) {
        auto stride_int = stride->as_int();
        if (stride_int == 1 || stride_int == -1) {
            // Convert condition to CNF for processing
            symbolic::CNF cnf;
            try {
                cnf = symbolic::conjunctive_normal_form(condition);
            } catch (...) {
                cnf = {{condition}};
            }

            // Build a new CNF with converted literals
            symbolic::CNF new_cnf;

            for (const auto& clause : cnf) {
                std::vector<symbolic::Condition> new_clause;

                for (const auto& literal : clause) {
                    if (!SymEngine::is_a<SymEngine::Unequality>(*literal)) {
                        // Keep non-unequality literals as-is
                        new_clause.push_back(literal);
                        continue;
                    }

                    auto ne = SymEngine::rcp_static_cast<const SymEngine::Unequality>(literal);
                    auto lhs = ne->get_arg1();
                    auto rhs = ne->get_arg2();

                    bool lhs_has_indvar = symbolic::uses(lhs, indvar->get_name());
                    bool rhs_has_indvar = symbolic::uses(rhs, indvar->get_name());

                    if (!lhs_has_indvar && !rhs_has_indvar) {
                        new_clause.push_back(literal);
                        continue;
                    }

                    if (lhs_has_indvar && rhs_has_indvar) {
                        new_clause.push_back(literal);
                        continue;
                    }

                    // Ensure indvar is on LHS
                    if (!lhs_has_indvar) {
                        std::swap(lhs, rhs);
                    }

                    // Check if affine with coefficient = 1
                    symbolic::SymbolVec syms = {indvar};
                    auto poly = symbolic::polynomial(lhs, syms);
                    if (poly.is_null()) {
                        new_clause.push_back(literal);
                        continue;
                    }

                    auto coeffs = symbolic::affine_coefficients(poly);
                    if (coeffs.empty() || coeffs.find(indvar) == coeffs.end()) {
                        new_clause.push_back(literal);
                        continue;
                    }

                    auto coeff = coeffs.at(indvar);
                    if (!SymEngine::is_a<SymEngine::Integer>(*coeff)) {
                        new_clause.push_back(literal);
                        continue;
                    }
                    auto coeff_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(coeff)->as_int();
                    if (coeff_int != 1) {
                        new_clause.push_back(literal);
                        continue;
                    }

                    // At this point, indvar should already be isolated by step 2
                    // Just convert != to </>
                    // Convert: stride > 0 → i < bound, stride < 0 → i > bound
                    symbolic::Condition new_literal;
                    if (stride_int > 0) {
                        new_literal = symbolic::Lt(indvar, rhs);
                    } else {
                        new_literal = symbolic::Gt(indvar, rhs);
                    }
                    new_clause.push_back(new_literal);
                }

                new_cnf.push_back(new_clause);
            }

            // Reconstruct condition from CNF
            condition = symbolic::__true__();
            for (const auto& clause : new_cnf) {
                if (clause.empty()) {
                    continue;
                }
                symbolic::Condition clause_cond = clause[0];
                for (size_t i = 1; i < clause.size(); ++i) {
                    clause_cond = symbolic::Or(clause_cond, clause[i]);
                }
                condition = symbolic::And(condition, clause_cond);
            }
        }
    }

    // Step 4: Post-normalization - simplify max(init, bound) → bound
    condition = simplify_max_bounds(condition, init);

    // Update the loop condition
    builder.update_loop(loop_, indvar, condition, init, loop_.update());
}

void LoopConditionNormalize::to_json(nlohmann::json& j) const {
    std::string loop_type = "for";
    if (dynamic_cast<const structured_control_flow::Map*>(&loop_)) {
        loop_type = "map";
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", loop_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = nlohmann::json::object();
}

LoopConditionNormalize LoopConditionNormalize::
    from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();

    auto element = builder.find_element_by_id(loop_id);
    if (element == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(loop_id) + " not found.");
    }

    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(element);
    if (loop == nullptr) {
        throw std::runtime_error("Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop.");
    }

    return LoopConditionNormalize(*loop);
}

} // namespace transformations
} // namespace sdfg
