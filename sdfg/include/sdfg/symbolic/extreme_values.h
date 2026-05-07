/**
 * @file extreme_values.h
 * @brief Computing bounds of symbolic expressions using assumptions
 *
 * This file provides the BoundAnalysis class and convenience functions for computing
 * the minimum and maximum values that symbolic expressions can take, given a set of
 * assumptions about symbol bounds.
 *
 * ## Bound Computation
 *
 * The bound computation uses assumptions about symbols (from assumptions.h) to determine
 * the extreme values (minimum and maximum) that an expression can reach. The computation
 * considers:
 * - Symbol bounds from assumptions
 * - Parameter symbols (treated as unknowns preserved in the result)
 * - Tight vs. loose bounds (exact vs. conservative estimates)
 *
 * ## Mathematical Foundation
 *
 * For an expression e(x, p) where x_i in [l_i, u_i] and p are free parameters:
 * - Polynomial normalization groups correlated addends (e.g. -2x + Tx -> (T-2)x)
 * - Sign-aware affine analysis uses monotonicity: if coeff > 0, min(c*x) = c*min(x)
 * - For multilinear products, extrema occur at vertices of the bounding box
 * - min(max(a,b)) = max(min(a), min(b)); max(min(a,b)) = min(max(a), max(b))
 *
 * @see assumptions.h for information about symbol assumptions
 * @see symbolic.h for building symbolic expressions
 */

#pragma once

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

/**
 * @brief Result of bounding an expression: an interval [lower, upper]
 *
 * Both endpoints are expressions in terms of parameters. A null endpoint
 * indicates that the bound could not be computed for that direction.
 */
struct Interval {
    Expression lower;
    Expression upper;

    /** @brief Returns true if both bounds failed */
    bool failed() const { return lower.is_null() && upper.is_null(); }

    /** @brief Returns true if the lower bound was computed */
    bool has_lower() const { return !lower.is_null(); }

    /** @brief Returns true if the upper bound was computed */
    bool has_upper() const { return !upper.is_null(); }

    /** @brief Creates an exact interval [expr, expr] for a constant or parameter */
    static Interval exact(const Expression& expr) { return {expr, expr}; }

    /** @brief Creates a failure result */
    static Interval failure() { return {SymEngine::null, SymEngine::null}; }
};

/**
 * @brief Computes bounds of symbolic expressions using assumptions
 *
 * BoundAnalysis walks the expression tree and computes both lower and upper bounds
 * simultaneously, using polynomial normalization and monotonicity analysis for
 * tight results.
 */
class BoundAnalysis {
public:
    BoundAnalysis(const SymbolSet& parameters, const Assumptions& assumptions, bool use_tight_assumptions);

    /** @brief Compute both lower and upper bounds of an expression */
    Interval bound(const Expression& expr);

    /** @brief Convenience: compute only the lower bound */
    Expression lower_bound(const Expression& expr);

    /** @brief Convenience: compute only the upper bound */
    Expression upper_bound(const Expression& expr);

private:
    static constexpr size_t MAX_DEPTH = 100;
    static constexpr size_t MAX_GENERATORS = 16;

    const SymbolSet& parameters_;
    const Assumptions& assumptions_;
    bool use_tight_;

    // Cycle detection: symbols currently being bounded
    SymbolSet visiting_;

    Interval visit(const Expression& expr, size_t depth);

    Interval visit_symbol(const SymEngine::RCP<const SymEngine::Symbol>& sym, size_t depth);
    Interval visit_function(const SymEngine::RCP<const SymEngine::FunctionSymbol>& func, size_t depth);
    Interval visit_pow(const SymEngine::RCP<const SymEngine::Pow>& pow_expr, size_t depth);
    Interval visit_mul(const SymEngine::RCP<const SymEngine::Mul>& mul_expr, size_t depth);
    Interval visit_add(const SymEngine::RCP<const SymEngine::Add>& add_expr, size_t depth);
    Interval visit_max(const Expression& expr, size_t depth);
    Interval visit_min(const Expression& expr, size_t depth);

    // Add sub-strategies
    Interval visit_add_affine(const AffineCoeffs& coeffs, const SymbolVec& gens, size_t depth);
    Interval visit_add_polynomial(const Polynomial& poly, const SymbolVec& gens, size_t depth);
    Interval visit_add_argwise(const SymEngine::vec_basic& args, size_t depth);
};

// ---- Backward-compatible free functions ----

/**
 * @brief Compute the minimum of an expression
 * @param expr The expression to compute the minimum of
 * @param parameters Symbols treated as free parameters in the result
 * @param assumptions Bounds information for symbols
 * @param tight If true, use tight (exact) bounds; if false, use loose (conservative) bounds
 * @return The minimum of the expression, or null if unbounded
 */
Expression minimum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight);

/**
 * @brief Compute the maximum of an expression
 * @param expr The expression to compute the maximum of
 * @param parameters Symbols treated as free parameters in the result
 * @param assumptions Bounds information for symbols
 * @param tight If true, use tight (exact) bounds; if false, use loose (conservative) bounds
 * @return The maximum of the expression, or null if unbounded
 */
Expression maximum(const Expression expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight);

// ---- Sound inequality proofs ----
//
// All `is_*` predicates return TRUE only when the inequality is provable from
// the supplied assumptions; FALSE means "unknown OR disproven" — callers must
// not interpret a false result as the negation of the predicate.
//
// Proof strategy (in order, all sound):
//   1. Direct: `symbolic::is_true(Op(a, b))` for trivially decidable cases.
//   2. Interval: compute `lower(a-b)` / `upper(a-b)` via `BoundAnalysis` and
//      compare against zero.
//   3. Min/Max descent: when the residue contains a `Min`/`Max` subexpression
//      in a monotone-nondecreasing position (e.g. additive top-level), recurse
//      on each replacement that yields a sound bound on the residue.
//
// Affine expressions over symbols with assumption-derived bounds are the
// expected input shape (subscript indices, stride differences). The descent
// is sound for these; for arbitrary expressions the interval step is the
// guaranteed-sound floor.

/** @brief Prove `expr >= 0`. */
bool is_nonneg(const Expression& expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight = false);

/** @brief Prove `expr > 0`. */
bool is_positive(const Expression& expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight = false);

/** @brief Prove `expr <= 0`. */
bool is_nonpos(const Expression& expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight = false);

/** @brief Prove `expr < 0`. */
bool is_negative(const Expression& expr, const SymbolSet& parameters, const Assumptions& assumptions, bool tight = false);

/** @brief Prove `a >= b`. */
bool is_ge(
    const Expression& a,
    const Expression& b,
    const SymbolSet& parameters,
    const Assumptions& assumptions,
    bool tight = false
);

/** @brief Prove `a > b`. */
bool is_gt(
    const Expression& a,
    const Expression& b,
    const SymbolSet& parameters,
    const Assumptions& assumptions,
    bool tight = false
);

/** @brief Prove `a <= b`. */
bool is_le(
    const Expression& a,
    const Expression& b,
    const SymbolSet& parameters,
    const Assumptions& assumptions,
    bool tight = false
);

/** @brief Prove `a < b`. */
bool is_lt(
    const Expression& a,
    const Expression& b,
    const SymbolSet& parameters,
    const Assumptions& assumptions,
    bool tight = false
);

/** @brief Prove `a == b` (both `a >= b` and `a <= b`). */
bool is_eq(
    const Expression& a,
    const Expression& b,
    const SymbolSet& parameters,
    const Assumptions& assumptions,
    bool tight = false
);

} // namespace symbolic
} // namespace sdfg
