#include "sdfg/symbolic/delinearization.h"

#include <algorithm>
#include <optional>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace symbolic {

namespace {

// A multiplicity-aware complexity score for stride expressions.
// Unlike atoms(), this accounts for repeated symbol use (e.g., s*s or s**2).
size_t stride_complexity_score(const sdfg::symbolic::Expression& expr) {
    if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
        return 0;
    }
    if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
        return 1;
    }
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow_expr = SymEngine::rcp_static_cast<const SymEngine::Pow>(expr);
        auto args = pow_expr->get_args();
        if (args.size() == 2 && SymEngine::is_a<SymEngine::Integer>(*args[1])) {
            try {
                long long exp = SymEngine::rcp_static_cast<const SymEngine::Integer>(args[1])->as_int();
                if (exp >= 0) {
                    return static_cast<size_t>(exp) * stride_complexity_score(args[0]);
                }
            } catch (const SymEngine::SymEngineException&) {
                return 0;
            }
        }
    }
    if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
        size_t score = 0;
        for (const auto& arg : SymEngine::rcp_static_cast<const SymEngine::Mul>(expr)->get_args()) {
            score += stride_complexity_score(arg);
        }
        return score;
    }
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        size_t score = 0;
        for (const auto& arg : SymEngine::rcp_static_cast<const SymEngine::Add>(expr)->get_args()) {
            score = std::max(score, stride_complexity_score(arg));
        }
        return score;
    }

    // Generic function / composite fallback: recurse through args and add a bonus
    // so function-bearing expressions are preferred over equally-scored plain ones.
    size_t score = 0;
    for (const auto& arg : expr->get_args()) {
        score += stride_complexity_score(arg);
    }
    if (SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
        score += 1;
    }
    return score;
}

bool provably_ge(const sdfg::symbolic::Expression& lhs, const sdfg::symbolic::Expression& rhs) {
    return sdfg::symbolic::is_true(sdfg::symbolic::Ge(lhs, rhs));
}

bool provably_gt(const sdfg::symbolic::Expression& lhs, const sdfg::symbolic::Expression& rhs) {
    return provably_ge(lhs, rhs) && !provably_ge(rhs, lhs);
}

// Decompose `expr` (linearized affine address) into a list of (stride, index)
// groups by parameter-monomial. Each term of `expr` is split into a parameter
// part (forming the "stride") and an indvar part (forming the "index"); terms
// that share the same stride monomial are merged by summing their indices.
//
// `params` is the set of symbols treated as parameters (constants wrt the
// access). Anything not in `params` is treated as an indvar contribution.
//
// Pure-parameter terms (no indvar atoms) are accumulated into `constant_offset`
// and never form their own group.
//
// Examples (params = {N}):
//   i*N + k          -> {N: i, 1: k}
//   i*N + i + j2     -> {N: i, 1: i + j2}
//   k2*N + i + j2    -> {N: k2, 1: i + j2}
//   j*N + j          -> {N: j, 1: j}
//
// Returns false if any factor mixes parameters and indvars in a way the
// simple split cannot handle (currently: never -- composite factors are
// classified as "indvar-side" if they touch any indvar atom).
bool decompose_by_stride(
    const sdfg::symbolic::Expression& expr,
    const sdfg::symbolic::SymbolSet& params,
    std::vector<std::pair<sdfg::symbolic::Expression, sdfg::symbolic::Expression>>& groups,
    sdfg::symbolic::Expression& constant_offset
) {
    namespace sym = sdfg::symbolic;
    constant_offset = sym::zero();
    // Walk the top-level Add structure WITHOUT expanding terms. This preserves
    // intentional factored layouts like `(B+2) * (i+1)` (one dim, stride B+2,
    // index i+1) while still letting same-indvar repetitions across distinct
    // additive terms (e.g. `j*N + j`) decompose into multiple groups.
    SymEngine::vec_basic terms;
    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        for (const auto& a : expr->get_args()) terms.push_back(a);
    } else {
        terms.push_back(expr);
    }

    auto add_to_group = [&](const sym::Expression& stride, const sym::Expression& index) {
        for (auto& g : groups) {
            if (sym::eq(g.first, stride)) {
                g.second = sym::add(g.second, index);
                return;
            }
        }
        groups.push_back({stride, index});
    };

    for (auto& term : terms) {
        // Pure-constant or pure-parameter term -> constant offset.
        bool has_indvar = false;
        for (auto& s : sym::atoms(term)) {
            if (params.count(s) == 0) {
                has_indvar = true;
                break;
            }
        }
        if (!has_indvar) {
            constant_offset = sym::add(constant_offset, term);
            continue;
        }

        // Split term's factors into (stride, index).
        sym::Expression stride = sym::one();
        sym::Expression index = sym::one();

        SymEngine::vec_basic factors;
        if (SymEngine::is_a<SymEngine::Mul>(*term)) {
            for (const auto& f : term->get_args()) factors.push_back(f);
        } else {
            factors.push_back(term);
        }

        for (auto& f : factors) {
            bool factor_has_indvar = false;
            for (auto& s : sym::atoms(f)) {
                if (params.count(s) == 0) {
                    factor_has_indvar = true;
                    break;
                }
            }
            sym::Expression fe = f;
            if (factor_has_indvar) {
                index = sym::eq(index, sym::one()) ? fe : sym::mul(index, fe);
            } else {
                stride = sym::eq(stride, sym::one()) ? fe : sym::mul(stride, fe);
            }
        }

        add_to_group(stride, index);
    }
    return true;
}

// Principled affine delinearization.
//
// Instead of greedily peeling terms and validating with form-sensitive
// heuristics, this reconstructs the multi-dimensional structure
// deterministically and then *verifies* the result:
//
//   1. Decompose into (stride, index) groups by *parameter monomial* (via the
//      shared `decompose_by_stride`), plus a scalar offset. Grouping at the
//      term level keeps an induction variable that appears at several strides
//      correctly split -- e.g. `i*N + i + j2` -> {N: i, 1: i + j2}.
//   2. Sign-normalize: pull the numeric coefficient (including sign) out of
//      each stride so strides are positive parameter monomials and the sign
//      lives in the index. This makes reverse/descending sweeps (`(-1, j)` ->
//      `(1, -j)`) first-class, and rejects non-affine (indvar*indvar) indices.
//   3. Divisibility chain: sort strides so each divides the previous
//      (row-major requires `s_{m-1} | ... | s_0`); the consecutive ratios are
//      the recovered dimension sizes. Incomparable strides -> decline.
//   4. Offset carry: distribute the scalar offset across the strides via
//      mixed-radix polynomial division (outer to inner). Residual must vanish.
//   5. Verify (don't trust): re-linearize and require symbolic equality to the
//      input, plus each index provably >= 0. Upper bounds are left optimistic
//      -- within the loop body the assumptions are exactly the loop ranges, so
//      the access is in-bounds by construction (the classic "optimistic
//      delinearization" premise).
//
// Returns std::nullopt to *decline* (non-affine indices, non-positive stride,
// non-chain strides, ...), in which case the caller falls back to the
// heuristic peeler. It never returns an unverified decomposition.
std::optional<sdfg::symbolic::DelinearizeResult> delinearize_affine(
    const sdfg::symbolic::Expression& dim,
    sdfg::symbolic::AssumptionsBounds& bounds,
    const sdfg::symbolic::SymbolSet& indvars_set,
    const sdfg::symbolic::SymbolSet& params_set
) {
    namespace sym = sdfg::symbolic;
    const sym::Assumptions& assums = bounds.assums();

    auto is_zero = [](const sym::Expression& e) { return sym::eq(sym::simplify(sym::expand(e)), sym::zero()); };
    // Sign / non-negativity proofs must keep the parameters (`N`, `_s0`, ...)
    // symbolic so coupled bounds cancel (e.g. `upper(j) = N - 3` makes
    // `N - 2 - j >= 1`). The `AssumptionsBounds` BoundAnalysis is built with
    // empty parameters, which loses that cancellation, so route these through
    // the assumptions-based overload with the real parameter set.
    auto stride_ge_one = [&](const sym::Expression& e) {
        return sym::is_ge(e, sym::one(), params_set, assums, /*tight=*/false);
    };
    auto index_nonneg = [&](const sym::Expression& e) {
        return sym::is_nonneg(e, params_set, assums, /*tight=*/false);
    };
    auto index_negative = [&](const sym::Expression& e) {
        return sym::is_negative(e, params_set, assums, /*tight=*/false);
    };

    // 1. Decompose into (stride, index) groups plus a scalar offset. This
    //    groups by *parameter monomial* at the term level (reusing the peeler's
    //    decomposition), which correctly keeps an induction variable that
    //    appears at multiple strides -- e.g. `i*N + i + j2` splits as
    //    {N: i, 1: i + j2} rather than collapsing `coeff(i) = N + 1` into a
    //    single bogus stride.
    std::vector<std::pair<sym::Expression, sym::Expression>> raw_groups;
    sym::Expression offset;
    if (!decompose_by_stride(dim, params_set, raw_groups, offset)) {
        return std::nullopt;
    }

    // 2. Normalize each group so the stride is *positive*: flip only the sign
    //    out of the stride into the index (the stride magnitude -- a constant
    //    like `20` or a parametric monomial like `4*_s0` -- is a legitimate
    //    row-major stride and must be preserved). This turns a reverse
    //    contribution `(-1, j)` into `(1, -j)` and `(-4*_s0, i)` into
    //    `(4*_s0, -i)`, making descending/reverse sweeps first-class, and
    //    rejects non-affine (indvar*indvar) indices.
    auto normalize_sign = [](const sym::Expression& s) -> std::pair<sym::Expression, sym::Expression> {
        bool neg = false;
        if (SymEngine::is_a<SymEngine::Mul>(*s)) {
            neg = SymEngine::rcp_static_cast<const SymEngine::Mul>(s)->get_coef()->is_negative();
        } else if (SymEngine::is_a_Number(*s)) {
            neg = SymEngine::rcp_static_cast<const SymEngine::Number>(s)->is_negative();
        }
        if (neg) {
            // Return {sign, |stride|}.
            return {sym::integer(-1), sym::mul(sym::integer(-1), s)};
        }
        return {sym::one(), s};
    };
    auto is_affine_index = [&](const sym::Expression& idx) -> bool {
        sym::SymbolVec iv(indvars_set.begin(), indvars_set.end());
        auto p = sym::polynomial(idx, iv);
        if (p.is_null()) return false;
        return !sym::affine_coefficients(p).empty();
    };

    struct DimGroup {
        sym::Expression stride;
        sym::Expression index; // affine combination of induction variables
    };
    std::vector<DimGroup> groups;
    for (auto& [raw_stride, raw_index] : raw_groups) {
        auto [sign, stride] = normalize_sign(raw_stride);
        sym::Expression index = sym::simplify(sym::expand(sym::mul(sign, raw_index)));
        if (sym::eq(index, sym::zero())) continue;
        if (!is_affine_index(index)) {
            return std::nullopt; // product of induction variables -> not row-major affine
        }
        bool merged = false;
        for (auto& g : groups) {
            if (sym::eq(g.stride, stride)) {
                g.index = sym::simplify(sym::expand(sym::add(g.index, index)));
                merged = true;
                break;
            }
        }
        if (!merged) groups.push_back({stride, index});
    }
    if (groups.empty()) {
        return std::nullopt;
    }

    // 3. Every stride must be a provably-positive parameter monomial (>= 1).
    for (auto& g : groups) {
        if (!stride_ge_one(g.stride)) {
            return std::nullopt;
        }
    }

    // 3b. Ensure a contiguous (stride-1) innermost dimension exists. Accesses
    //     whose innermost index is a compile-time constant -- e.g. a fixed
    //     column `A[_s0*(1+i)]` (col 0) or `A[_s0*(1+i) + N - 1]` (col N-1),
    //     the ADI Dirichlet boundary writes -- produce only strided groups and
    //     no stride-1 group. Inject a synthetic contiguous dimension with a
    //     constant index; the offset carry (step 6) populates it with the
    //     residual column offset, recovering `[row, const-col]` in a row-major
    //     layout. Sound: verified by the re-linearization in step 7.
    //
    //     Only do this when a group stride is *parametric* (e.g. `_s0`): a
    //     parametric stride is a real array leading dimension, so the access is
    //     a column of a multi-dim array. A purely constant stride (e.g. `2*j`)
    //     is a strided 1D access into a 1D array and must stay 1D -- injecting
    //     there would wrongly report it as 2D and break dimension-matching
    //     consumers (e.g. MapFusion).
    bool has_unit_stride = false;
    bool has_parametric_stride = false;
    for (auto& g : groups) {
        if (sym::eq(g.stride, sym::one())) has_unit_stride = true;
        if (!SymEngine::is_a<SymEngine::Integer>(*g.stride)) has_parametric_stride = true;
    }
    if (!has_unit_stride && has_parametric_stride) {
        groups.push_back({sym::one(), sym::zero()});
    }

    // 4. Sort into a divisibility chain (outermost/largest stride first).
    //    `a` is outer than `b` iff `b.stride` divides `a.stride`.
    auto divides = [&](const sym::Expression& a, const sym::Expression& b) -> bool {
        // Does `a` divide `b`? i.e. `b mod a == 0`.
        auto [q, r] = sym::polynomial_div(b, a);
        return is_zero(r);
    };
    std::stable_sort(groups.begin(), groups.end(), [&](const DimGroup& A, const DimGroup& B) {
        if (sym::eq(A.stride, B.stride)) return false;
        if (divides(B.stride, A.stride)) return true; // A is a multiple of B -> A outer
        if (divides(A.stride, B.stride)) return false; // B is a multiple of A -> B outer
        return false; // incomparable -> caught below
    });

    // 5. Validate the chain and that the innermost stride is 1.
    const size_t m = groups.size();
    for (size_t t = 0; t + 1 < m; ++t) {
        auto [q, r] = sym::polynomial_div(groups[t].stride, groups[t + 1].stride);
        if (!is_zero(r)) {
            return std::nullopt; // incomparable / not a clean row-major chain
        }
    }
    if (!sym::eq(groups[m - 1].stride, sym::one())) {
        return std::nullopt; // innermost dimension is not contiguous
    }

    // 6. Distribute the scalar offset across the strides (outer to inner) via
    //    mixed-radix polynomial division.
    sym::Expression carry = offset;
    for (size_t t = 0; t < m; ++t) {
        auto [q, r] = sym::polynomial_div(carry, groups[t].stride);
        groups[t].index = sym::simplify(sym::expand(sym::add(groups[t].index, q)));
        carry = r;
    }
    if (!is_zero(carry)) {
        return std::nullopt; // constant did not decompose onto the strides
    }

    // 6b. Borrow: the truncated polynomial division above can leave an inner
    //     index provably negative -- e.g. a negative column offset
    //     `_s1*(1+i) - 1` distributes to `[i+1, -1]`. The borrow
    //     `index[t] += d_t; index[t-1] -= 1` (with `d_t = stride[t-1]/stride[t]`
    //     the size of dimension t) is value-preserving because
    //     `d_t*stride[t] == stride[t-1]`, so it rewrites into the canonical
    //     row-major form `[.., row-1, col + d_t]` -> `[i, _s1 - 1]` for the
    //     deriche boundary accesses. Iterate inner-to-outer; a small cap guards
    //     against non-terminating symbolic cases.
    for (size_t t = m; t-- > 1;) {
        auto [d, r] = sym::polynomial_div(groups[t - 1].stride, groups[t].stride);
        if (!is_zero(r)) continue; // not a clean ratio (shouldn't happen after step 5)
        int guard = 0;
        while (index_negative(groups[t].index) && guard++ < 64) {
            groups[t].index = sym::simplify(sym::expand(sym::add(groups[t].index, d)));
            groups[t - 1].index = sym::simplify(sym::expand(sym::sub(groups[t - 1].index, sym::one())));
        }
    }

    // 7. Structural verification: re-linearize and require equality.
    sym::Expression recon = sym::zero();
    for (size_t t = 0; t < m; ++t) {
        recon = sym::add(recon, sym::mul(groups[t].stride, groups[t].index));
    }
    if (!is_zero(sym::sub(dim, recon))) {
        return std::nullopt; // proposal is not algebraically identical -> never trust
    }

    // 8. Index sanity: every recovered index must be provably non-negative
    //    (guarantees non-negative offsets downstream). Upper bounds are left
    //    optimistic per the loop-body-containment premise.
    //
    //    Skipped for a single-dimension (m == 1) result: there the index is
    //    just the raw linear offset into an unbounded 1D region -- there is no
    //    row-major sub-structure whose bounds a negative index could violate.
    //    This admits reverse/triangular 1D accesses like `k - i0` (durbin).
    if (m > 1) {
        for (size_t t = 0; t < m; ++t) {
            if (!index_nonneg(groups[t].index)) {
                return std::nullopt;
            }
        }
    }

    // Assemble the result: indices per dimension (outer to inner) and the
    // recovered dimension sizes (consecutive stride ratios). The leading
    // dimension is left unbounded by the caller.
    sym::DelinearizeResult result;
    for (size_t t = 0; t < m; ++t) {
        result.indices.push_back(groups[t].index);
    }
    for (size_t t = 0; t + 1 < m; ++t) {
        auto [q, r] = sym::polynomial_div(groups[t].stride, groups[t + 1].stride);
        result.dimensions.push_back(sym::simplify(q));
    }
    result.success = true;
    return result;
}

} // namespace

DelinearizeResult delinearize(const Expression& expr, AssumptionsBounds& bounds) {
    auto dim = expr;
    const Assumptions& assums = bounds.assums();

    // Hoisted BoundAnalysis references from the caller-supplied bundle: every
    // internal lower/upper bound query in this function shares the bundle's
    // memoization cache, which amortizes across all delinearize calls that
    // pass the same `bounds`.
    BoundAnalysis& ba_loose = bounds.loose();
    BoundAnalysis& ba_tight = bounds.tight();

    // Partition atoms into indvars (non-constant in `assums`) and parameters
    // (constant or unknown).
    SymbolSet indvars_set;
    SymbolSet params_set;
    for (auto& sym : atoms(dim)) {
        auto it = assums.find(sym);
        if (it != assums.end() && (!it->second.constant() || !it->second.map().is_null())) {
            indvars_set.insert(sym);
        } else {
            params_set.insert(sym);
        }
    }

    // Trivial: no indvar -> single-dim with the expression as its (constant) index.
    if (indvars_set.empty()) {
        return {MultiExpression{dim}, MultiExpression{}, true};
    }

    // Principled affine path: normal-form + divisibility-chain + verify. This
    // is form-independent and handles signed / reverse indices. It declines
    // (returns nullopt) for non-affine or ambiguous cases, in which case we
    // fall back to the heuristic peeler below.
    if (auto affine = delinearize_affine(dim, bounds, indvars_set, params_set)) {
        return *affine;
    }

    // Step 1: Decompose into stride-monomial groups.
    std::vector<std::pair<Expression, Expression>> groups;
    Expression offset;
    if (!decompose_by_stride(dim, params_set, groups, offset)) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }
    if (groups.empty()) {
        // Only a constant offset -> not a real access; fall through as 1D.
        return {MultiExpression{dim}, MultiExpression{}, true};
    }

    // Step 1a: Normalize each group's index by extracting params-only addends
    // into the global offset. SymEngine canonicalizes `224 * (2*hout + kh - 3)`
    // into a single Mul `(224, -3+2*hout+kh)` whose index `-3+2*hout+kh` is
    // negative in the lower corner — `is_nonneg` then refuses to peel it.
    // Splitting `-3` off into `offset += 224 * -3` leaves index `2*hout + kh`
    // which IS provably non-negative.
    //
    // Soundness: `stride * (idx_part + const_part) == stride * idx_part +
    // stride * const_part`, and addends that touch no indvar are constants
    // relative to the access pattern and naturally belong in the offset.
    for (auto& [stride, index] : groups) {
        if (!SymEngine::is_a<SymEngine::Add>(*index)) continue;
        Expression idx_part = symbolic::zero();
        Expression const_part = symbolic::zero();
        for (const auto& addend : index->get_args()) {
            bool has_indvar = false;
            for (const auto& s : symbolic::atoms(addend)) {
                if (params_set.count(s) == 0) {
                    has_indvar = true;
                    break;
                }
            }
            if (has_indvar) {
                idx_part = symbolic::eq(idx_part, symbolic::zero()) ? Expression(addend)
                                                                    : symbolic::add(idx_part, addend);
            } else {
                const_part = symbolic::eq(const_part, symbolic::zero()) ? Expression(addend)
                                                                        : symbolic::add(const_part, addend);
            }
        }
        if (!symbolic::eq(const_part, symbolic::zero())) {
            offset = symbolic::add(offset, symbolic::mul(stride, const_part));
            index = idx_part;
        }
    }
    // Drop groups whose index collapsed to zero (purely-constant contribution
    // already accounted for in the offset).
    groups.erase(
        std::remove_if(
            groups.begin(), groups.end(), [](const auto& g) { return symbolic::eq(g.second, symbolic::zero()); }
        ),
        groups.end()
    );
    if (groups.empty()) {
        return {MultiExpression{dim}, MultiExpression{}, true};
    }

    // Step 1b: Merge sibling groups sharing an index when at least one of them
    // has a stride that cannot stand alone as a leading dimension stride
    // (i.e. its provable lower bound is not >= 1, including symbolic strides
    // like `-4*_s0`). Such non-free-standing strides represent additive
    // contributions to a parametric stride sharing the same index. This
    // recovers expanded forms like `_s0^2*i - 4*_s0*i + 4*i + _s0*j - 2*j + k`
    // into the 3-dim shape `(_s0-2)^2*i + (_s0-2)*j + k`. Cases like
    // `j*N + j` (strides {N, 1}, both >= 1) remain unmerged as 2 dims.
    auto stride_is_free_standing = [&](const Expression& s) -> bool {
        auto lb = ba_loose.lower_bound(s);
        if (lb == SymEngine::null) return false;
        return symbolic::is_true(symbolic::Ge(lb, symbolic::one()));
    };
    {
        // Collect indices that have any non-free-standing sibling.
        std::vector<Expression> merge_indices;
        for (size_t a = 0; a < groups.size(); ++a) {
            if (stride_is_free_standing(groups[a].first)) continue;
            // Skip if no sibling shares this index.
            bool has_sibling = false;
            for (size_t b = 0; b < groups.size(); ++b) {
                if (b != a && symbolic::eq(groups[b].second, groups[a].second)) {
                    has_sibling = true;
                    break;
                }
            }
            if (!has_sibling) continue;
            bool already = false;
            for (auto& mi : merge_indices) {
                if (symbolic::eq(mi, groups[a].second)) {
                    already = true;
                    break;
                }
            }
            if (!already) merge_indices.push_back(groups[a].second);
        }
        // For each such index, sum all groups with that index into one.
        for (const auto& idx : merge_indices) {
            Expression combined = SymEngine::null;
            for (size_t k = groups.size(); k-- > 0;) {
                if (symbolic::eq(groups[k].second, idx)) {
                    combined = combined.is_null() ? groups[k].first : symbolic::add(groups[k].first, combined);
                    groups.erase(groups.begin() + k);
                }
            }
            if (!combined.is_null()) {
                // Factor the merged stride to recover patterns like
                // `_s0^2 - 4*_s0 + 4` -> `(_s0-2)^2` so that bound analysis
                // (`minimum`) can prove it's >= 1.
                groups.push_back({symbolic::factor(symbolic::expand(combined)), idx});
            }
        }
    }

    // Step 2: Peel-off groups in dominance order.
    DelinearizeResult result;
    MultiExpression strides; // strides in peel order
    Expression remaining = symbolic::sub(dim, offset);

    while (!groups.empty()) {
        // Snapshot `remaining` so that the on-failure merge fallback below
        // can backtrack the peel-attempt's subtraction and retry with a
        // different group decomposition.
        Expression saved_remaining = remaining;

        // Pick the group with the strongest stride using:
        // 1) provable bound dominance, 2) multiplicity-aware complexity,
        // 3) atom-count fallback. Same heuristic as before but keyed on
        //    the stride monomial rather than per-symbol affine coefficient.
        int best_idx = -1;
        Expression best_stride = SymEngine::null;
        Expression best_index = SymEngine::null;
        Expression best_lb = SymEngine::null;
        Expression best_ub = SymEngine::null;
        size_t best_complexity = 0;
        size_t max_atom_count = 0;

        for (size_t k = 0; k < groups.size(); ++k) {
            const auto& stride = groups[k].first;
            const auto& index = groups[k].second;
            auto lb = ba_loose.lower_bound(stride);
            auto ub = ba_loose.upper_bound(stride);
            size_t complexity = stride_complexity_score(stride);
            size_t atom_count = symbolic::atoms(stride).size();

            bool better = false;
            if (best_idx < 0) {
                better = true;
            } else {
                if (complexity > best_complexity) {
                    better = true;
                } else if (complexity == best_complexity) {
                    if (lb != SymEngine::null && best_lb != SymEngine::null && provably_gt(lb, best_lb)) {
                        better = true;
                    }
                    if (!better && ub != SymEngine::null && best_ub != SymEngine::null && provably_gt(ub, best_ub)) {
                        better = true;
                    }
                    if (!better && atom_count > max_atom_count) {
                        better = true;
                    }
                    if (!better && atom_count == max_atom_count) {
                        // Only break ties lexicographically when strides are truly
                        // indistinguishable: skip if the current best is strictly
                        // larger by lb/ub. Without this guard the lex tiebreak can
                        // overwrite an objectively larger stride (e.g. 7 vs 1) with
                        // a smaller one purely because of index-name ordering,
                        // breaking the dominance-order peel.
                        bool best_strictly_better =
                            (lb != SymEngine::null && best_lb != SymEngine::null && provably_gt(best_lb, lb)) ||
                            (ub != SymEngine::null && best_ub != SymEngine::null && provably_gt(best_ub, ub));
                        if (!best_strictly_better && SymEngine::str(*index) > SymEngine::str(*best_index)) {
                            better = true;
                        }
                    }
                }
            }

            if (better) {
                best_idx = static_cast<int>(k);
                best_stride = stride;
                best_index = index;
                best_lb = lb;
                best_ub = ub;
                best_complexity = complexity;
                max_atom_count = atom_count;
            }
        }
        if (best_idx < 0) {
            break;
        }

        // Index must be nonnegative under assumptions.
        if (!is_nonneg(best_index, ba_loose)) {
            break;
        }

        // Stride must be positive.
        Expression stride = best_stride;
        auto stride_lb = best_lb;
        if (stride_lb == SymEngine::null) {
            stride_lb = ba_loose.lower_bound(stride);
        }
        if (stride_lb.is_null()) {
            break;
        }
        if (!symbolic::is_true(symbolic::Ge(stride_lb, symbolic::one()))) {
            break;
        }

        // Peel off the dimension contribution.
        remaining = symbolic::sub(remaining, symbolic::mul(stride, best_index));
        remaining = symbolic::expand(remaining);
        remaining = symbolic::simplify(remaining);

        // Remaining must be nonnegative.
        if (!is_nonneg(remaining, ba_loose)) {
            break;
        }

        // Remaining must be strictly less than this stride (so this peel is
        // unambiguous wrt the remaining lower-stride contributions).
        bool stride_check_passed = false;

        // Direct path: ask the symbolic engine to prove `stride > remaining`.
        if (symbolic::is_true(symbolic::Gt(stride, remaining))) {
            stride_check_passed = true;
        }

        // Substitution path: replace each indvar in `remaining` (not appearing
        // in `stride`) by its assumption-provided upper bound, then re-check.
        // This recovers coupled bounds like `i + j2 <= N - 1` from the
        // assumption `j2 <= N - i - 1` that BoundAnalysis doesn't synthesize.
        if (!stride_check_passed) {
            Expression r = remaining;
            SymbolSet stride_atoms = symbolic::atoms(stride);
            for (int iter = 0; iter < 16; ++iter) {
                bool changed = false;
                SymbolSet r_atoms = symbolic::atoms(r);
                for (auto& s : r_atoms) {
                    if (stride_atoms.count(s)) continue;
                    auto it = assums.find(s);
                    if (it == assums.end()) continue;
                    // Combine ALL known upper bounds (tight + listed) by taking
                    // their minimum. Using only tight_upper_bound() can drop
                    // important problem-specific bounds (e.g. j20 <= N-i-1
                    // when tight is the tile bound). Skip infinity placeholders.
                    auto is_finite = [](const Expression& b) {
                        return !b.is_null() && !SymEngine::is_a<SymEngine::Infty>(*b);
                    };
                    Expression ub_s = is_finite(it->second.tight_upper_bound()) ? it->second.tight_upper_bound()
                                                                                : Expression(SymEngine::null);
                    for (auto& b : it->second.upper_bounds()) {
                        if (!is_finite(b)) continue;
                        ub_s = ub_s.is_null() ? b : symbolic::min(ub_s, b);
                    }
                    if (ub_s.is_null()) continue;
                    Expression r_new = symbolic::simplify(symbolic::expand(symbolic::subs(r, s, ub_s)));
                    if (!symbolic::eq(r_new, r)) {
                        r = r_new;
                        changed = true;
                    }
                }
                if (!changed) break;
            }
            // Min-aware `is_gt` API helper proves `stride > min(...)` by
            // proving `stride > a_i` for some Min arg.
            if (!stride_check_passed) {
                if (is_gt(stride, r, ba_loose)) {
                    stride_check_passed = true;
                }
            }
        }

        auto ub_remaining = stride_check_passed ? Expression(SymEngine::null) : ba_tight.upper_bound(remaining);
        if (!stride_check_passed && ub_remaining.is_null()) {
            ub_remaining = ba_loose.upper_bound(remaining);
        }

        if (!stride_check_passed && ub_remaining != SymEngine::null) {
            std::vector<Expression> ub_candidates;
            if (SymEngine::is_a<SymEngine::Min>(*ub_remaining)) {
                for (const auto& arg : ub_remaining->get_args()) {
                    ub_candidates.push_back(arg);
                }
            } else {
                ub_candidates.push_back(ub_remaining);
            }

            for (const auto& ub_cand : ub_candidates) {
                auto diff = symbolic::expand(symbolic::sub(stride, ub_cand));
                if (SymEngine::is_a<SymEngine::Integer>(*diff)) {
                    auto int_val = SymEngine::rcp_static_cast<const SymEngine::Integer>(diff);
                    if (int_val->is_positive()) {
                        stride_check_passed = true;
                        break;
                    }
                }
                auto cond = symbolic::Gt(stride, ub_cand);
                if (symbolic::is_true(cond)) {
                    stride_check_passed = true;
                    break;
                }
            }

            if (!stride_check_passed) {
                auto ub_stride = (best_ub == SymEngine::null) ? ba_loose.upper_bound(stride) : best_ub;
                if (ub_stride != SymEngine::null) {
                    auto cond_stride = symbolic::Ge(ub_stride, ub_remaining);
                    if (symbolic::is_true(cond_stride)) {
                        stride_check_passed = true;
                    }
                }
            }
        }

        // Offset-aware fallback: the actual value contributing below `stride`
        // after this peel is `remaining + r`, where `(q, r) = polynomial_div(
        // offset, stride)` absorbs the floor/truncated quotient into the new
        // index.
        if (!stride_check_passed) {
            auto [q_pre, r_pre] = polynomial_div(offset, stride);
            auto access = symbolic::expand(symbolic::add(remaining, r_pre));
            if (is_nonneg(access, ba_loose) && is_gt(stride, access, ba_loose)) {
                stride_check_passed = true;
            }
        }

        // Sub-dominant fallback: if the stride check still fails AND there is
        // a strictly smaller integer-stride sibling whose stride divides
        // `best.stride`, merge `best` into the sibling: the sibling becomes
        //   (sibling.stride, (best.stride / sibling.stride) * best.index + sibling.index).
        if (!stride_check_passed) {
            int merge_target = -1;
            long long merge_factor = 0;
            if (SymEngine::is_a<SymEngine::Integer>(*best_stride)) {
                long long b_stride = SymEngine::rcp_static_cast<const SymEngine::Integer>(best_stride)->as_int();
                for (size_t k = 0; k < groups.size(); ++k) {
                    if (static_cast<int>(k) == best_idx) continue;
                    const auto& other = groups[k];
                    if (!SymEngine::is_a<SymEngine::Integer>(*other.first)) continue;
                    long long o_stride = SymEngine::rcp_static_cast<const SymEngine::Integer>(other.first)->as_int();
                    if (o_stride <= 0 || o_stride >= b_stride) continue;
                    if (b_stride % o_stride != 0) continue;
                    if (symbolic::eq(other.second, best_index)) continue;
                    merge_target = static_cast<int>(k);
                    merge_factor = b_stride / o_stride;
                    break;
                }
            }
            if (merge_target >= 0) {
                Expression merged_index = symbolic::expand(symbolic::add(
                    symbolic::mul(symbolic::integer(merge_factor), best_index), groups[merge_target].second
                ));
                groups[merge_target].second = merged_index;
                groups.erase(groups.begin() + best_idx);
                remaining = saved_remaining; // undo peel mutation
                continue;
            }
        }

        if (!stride_check_passed) {
            break;
        }

        // Add offset contribution of this dimension via polynomial division.
        auto [q, r] = polynomial_div(offset, stride);
        offset = r;
        auto final_dim = symbolic::add(best_index, q);

        result.indices.push_back(final_dim);
        strides.push_back(stride);
        groups.erase(groups.begin() + best_idx);
    }

    // Not all groups could be peeled off.
    if (!groups.empty()) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }

    // Offset did not reduce to zero.
    if (!symbolic::eq(offset, symbolic::zero())) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }

    if (strides.empty()) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }

    // Final stride must be 1 (contiguous innermost dimension).
    if (!symbolic::eq(strides.back(), symbolic::one())) {
        return {MultiExpression{dim}, MultiExpression{}, false};
    }

    // Convert strides to dimensions by dividing consecutive strides.
    for (size_t i = 0; i + 1 < strides.size(); ++i) {
        auto [q, r] = polynomial_div(strides[i], strides[i + 1]);
        if (!symbolic::eq(r, symbolic::zero())) {
            return {MultiExpression{dim}, MultiExpression{}, false};
        }
        result.dimensions.push_back(q);
    }

    result.success = true;
    return result;
}

DelinearizeResult delinearize(const Expression& expr, const Assumptions& assums) {
    AssumptionsBounds bounds(assums);
    return delinearize(expr, bounds);
}

} // namespace symbolic
} // namespace sdfg
