#include "sdfg/symbolic/delinearization.h"

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

} // namespace

DelinearizeResult delinearize(const Expression& expr, const Assumptions& assums) {
    auto dim = expr;

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

    // Step 1b: Merge sibling groups sharing an index when at least one of them
    // has a stride that cannot stand alone as a leading dimension stride
    // (i.e. its provable lower bound is not >= 1, including symbolic strides
    // like `-4*_s0`). Such non-free-standing strides represent additive
    // contributions to a parametric stride sharing the same index. This
    // recovers expanded forms like `_s0^2*i - 4*_s0*i + 4*i + _s0*j - 2*j + k`
    // into the 3-dim shape `(_s0-2)^2*i + (_s0-2)*j + k`. Cases like
    // `j*N + j` (strides {N, 1}, both >= 1) remain unmerged as 2 dims.
    auto stride_is_free_standing = [&](const Expression& s) -> bool {
        auto lb = minimum(s, {}, assums, false);
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
            auto lb = minimum(stride, {}, assums, false);
            auto ub = maximum(stride, {}, assums, false);
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
                    // Final deterministic tie-break (lexicographic on index expr's string form
                    // to give iteration-order-independent results).
                    if (!better && atom_count == max_atom_count) {
                        if (SymEngine::str(*index) > SymEngine::str(*best_index)) {
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
        if (!is_nonneg(best_index, {}, assums, false)) {
            break;
        }

        // Stride must be positive.
        Expression stride = best_stride;
        auto stride_lb = best_lb;
        if (stride_lb == SymEngine::null) {
            stride_lb = minimum(stride, {}, assums, false);
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
        if (!is_nonneg(remaining, {}, assums, false)) {
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
                if (is_gt(stride, r, {}, assums, false)) {
                    stride_check_passed = true;
                }
            }
        }

        auto ub_remaining = stride_check_passed ? Expression(SymEngine::null) : maximum(remaining, {}, assums, true);
        if (!stride_check_passed && ub_remaining.is_null()) {
            ub_remaining = maximum(remaining, {}, assums, false);
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
                auto ub_stride = (best_ub == SymEngine::null) ? maximum(stride, {}, assums, false) : best_ub;
                if (ub_stride != SymEngine::null) {
                    auto cond_stride = symbolic::Ge(ub_stride, ub_remaining);
                    if (symbolic::is_true(cond_stride)) {
                        stride_check_passed = true;
                    }
                }
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

} // namespace symbolic
} // namespace sdfg
