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

#include <unordered_map>

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
 *
 * The instance is the unit of reuse: construct once with the parameter set and
 * assumptions, then call `bound`/`lower_bound`/`upper_bound` for each expression.
 * Results are memoized internally, so repeated queries for the same expression
 * (or shared subterms across queries) return in O(1) hash lookup. The cache is
 * sound under the fixed `(parameters, assumptions, tight)` triple this instance
 * was constructed with; create a new instance to bound under different
 * assumptions.
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

    // Memoization cache for visit() results, keyed by canonical SymEngine
    // hash+eq on the input expression. Only fully-successful intervals
    // (both endpoints non-null) are cached: such results are derived purely
    // from assumption bounds and are independent of the call context.
    // Failures and one-sided intervals can arise from cycle detection in
    // `visit_symbol`, whose state is context-dependent and must not be
    // reused, so they are recomputed every time.
    struct BasicHash {
        size_t operator()(const Expression& e) const noexcept { return e->hash(); }
    };
    struct BasicEq {
        bool operator()(const Expression& a, const Expression& b) const noexcept {
            return a.get() == b.get() || SymEngine::eq(*a, *b);
        }
    };
    std::unordered_map<Expression, Interval, BasicHash, BasicEq> cache_;

    Interval visit(const Expression& expr, size_t depth);
    Interval visit_uncached(const Expression& expr, size_t depth);

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

// ---- BoundAnalysis-based overloads (cache-amortized) ----
//
// These overloads route all internal interval queries through the supplied
// `BoundAnalysis`, so the cache amortizes across many predicate calls under
// the same `(parameters, assumptions, tight)` triple. They omit the empty-
// parameter fallback that the legacy overloads perform when the caller's
// parameter set is non-empty; this fallback is a no-op for callers that
// already use empty parameters (e.g. `delinearize`) and matters only when
// proving inequalities about parameters themselves.

/** @brief Prove `expr >= 0` using a pre-built `BoundAnalysis`. */
bool is_nonneg(const Expression& expr, BoundAnalysis& ba);

/** @brief Prove `expr > 0` using a pre-built `BoundAnalysis`. */
bool is_positive(const Expression& expr, BoundAnalysis& ba);

/** @brief Prove `expr <= 0` using a pre-built `BoundAnalysis`. */
bool is_nonpos(const Expression& expr, BoundAnalysis& ba);

/** @brief Prove `expr < 0` using a pre-built `BoundAnalysis`. */
bool is_negative(const Expression& expr, BoundAnalysis& ba);

/** @brief Prove `a >= b` using a pre-built `BoundAnalysis`. */
bool is_ge(const Expression& a, const Expression& b, BoundAnalysis& ba);

/** @brief Prove `a > b` using a pre-built `BoundAnalysis`. */
bool is_gt(const Expression& a, const Expression& b, BoundAnalysis& ba);

/** @brief Prove `a <= b` using a pre-built `BoundAnalysis`. */
bool is_le(const Expression& a, const Expression& b, BoundAnalysis& ba);

/** @brief Prove `a < b` using a pre-built `BoundAnalysis`. */
bool is_lt(const Expression& a, const Expression& b, BoundAnalysis& ba);

/** @brief Prove `a == b` using a pre-built `BoundAnalysis`. */
bool is_eq(const Expression& a, const Expression& b, BoundAnalysis& ba);

/**
 * @brief Bundles `Assumptions` with both the loose and tight `BoundAnalysis`
 *        derived from them.
 *
 * Many symbolic operations (`delinearize`, `is_subset`, `is_disjoint`,
 * `dependence_deltas`) issue many internal `minimum`/`maximum` queries against
 * the same `Assumptions` while occasionally needing tight vs. loose bounds.
 * Constructing one `AssumptionsBounds` per scope and threading it through these
 * operations lets the `BoundAnalysis` memoization cache amortize across all of
 * them — replacing N fresh `BoundAnalysis` constructions and N independent
 * caches with a single pair per scope.
 *
 * The wrapper holds a reference to the underlying `Assumptions`, so callers
 * must keep the `Assumptions` alive for the lifetime of the `AssumptionsBounds`.
 */
class AssumptionsBounds {
public:
    explicit AssumptionsBounds(const Assumptions& assums)
        : assums_(assums), loose_(empty_params(), assums, /*tight=*/false),
          tight_(empty_params(), assums, /*tight=*/true) {}

    const Assumptions& assums() const { return assums_; }
    BoundAnalysis& loose() { return loose_; }
    BoundAnalysis& tight() { return tight_; }

private:
    static const SymbolSet& empty_params() {
        static const SymbolSet kEmpty;
        return kEmpty;
    }

    const Assumptions& assums_;
    BoundAnalysis loose_;
    BoundAnalysis tight_;
};

} // namespace symbolic
} // namespace sdfg
