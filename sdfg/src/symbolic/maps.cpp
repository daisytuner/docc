#include "sdfg/symbolic/maps.h"

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>
#include <isl/space.h>

#include "sdfg/symbolic/delinearization.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/polyhedral.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/utils.h"

namespace sdfg {
namespace symbolic {
namespace maps {

bool is_monotonic_affine(const Expression expr, const Symbol sym, const Assumptions& assums) {
    SymbolVec symbols = {sym};
    auto poly = polynomial(expr, symbols);
    if (poly == SymEngine::null) {
        return false;
    }
    auto coeffs = affine_coefficients(poly);
    if (coeffs.empty()) {
        return false;
    }
    auto mul = minimum(coeffs[sym], {}, assums, false);
    if (mul == SymEngine::null) {
        return false;
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*mul)) {
        return false;
    }
    auto mul_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(mul);
    try {
        long long val = mul_int->as_int();
        if (val <= 0) {
            return false;
        }
    } catch (const SymEngine::SymEngineException&) {
        return false;
    }

    return true;
}

bool is_monotonic_pow(const Expression expr, const Symbol sym, const Assumptions& assums) {
    if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
        auto pow = SymEngine::rcp_dynamic_cast<const SymEngine::Pow>(expr);
        auto base = pow->get_base();
        auto exp = pow->get_exp();
        if (SymEngine::is_a<SymEngine::Integer>(*exp) && SymEngine::is_a<SymEngine::Symbol>(*base)) {
            auto exp_int = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(exp);
            try {
                long long val = exp_int->as_int();
                if (val <= 0) {
                    return false;
                }
            } catch (const SymEngine::SymEngineException&) {
                return false;
            }
            auto base_sym = SymEngine::rcp_static_cast<const SymEngine::Symbol>(base);
            auto ub_sym = minimum(base_sym, {}, assums, false);
            if (ub_sym == SymEngine::null) {
                return false;
            }
            auto positive = symbolic::Ge(ub_sym, symbolic::integer(0));
            return symbolic::is_true(positive);
        }
    }

    return false;
}

bool is_monotonic(const Expression expr, const Symbol sym, const Assumptions& assums) {
    if (is_monotonic_affine(expr, sym, assums)) {
        return true;
    }
    return is_monotonic_pow(expr, sym, assums);
}

DependenceDeltas compute_deltas_isl(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    AssumptionsBounds& bounds1,
    AssumptionsBounds& bounds2
) {
    const Assumptions& assums1 = bounds1.assums();
    const Assumptions& assums2 = bounds2.assums();

    if (expr1.size() != expr2.size()) {
        return DependenceDeltas{false, "", {}};
    }
    if (expr1.empty()) {
        return DependenceDeltas{false, "", {}};
    }

    // Transform both expressions into two maps with separate dimensions
    auto expr1_delinearized = expr1;
    if (expr1.size() == 1) {
        auto result = symbolic::delinearize(expr1.at(0), bounds1);
        if (result.success) {
            expr1_delinearized = result.indices;
        }
    }
    auto expr2_delinearized = expr2;
    if (expr2.size() == 1) {
        auto result = symbolic::delinearize(expr2.at(0), bounds2);
        if (result.success) {
            expr2_delinearized = result.indices;
        }
    }

    if (expr1_delinearized.size() != expr2_delinearized.size()) {
        return DependenceDeltas{false, "", {}};
    }

    // Cheap per-dimension check — if any single dimension is provably disjoint, no dependence
    for (size_t i = 0; i < expr1_delinearized.size(); i++) {
        auto& dim1 = expr1_delinearized[i];
        auto& dim2 = expr2_delinearized[i];
        auto maps = expressions_to_intersection_map_str({dim1}, {dim2}, indvar, assums1, assums2);
        polyhedral::IslCtx ctx;
        if (!ctx) {
            continue;
        }

        polyhedral::IslMap map_1(isl_map_read_from_str(ctx.get(), std::get<0>(maps).c_str()));
        polyhedral::IslMap map_2(isl_map_read_from_str(ctx.get(), std::get<1>(maps).c_str()));
        if (!map_1 || !map_2) {
            continue;
        }

        // Per-dimension disjointness: check if the RANGES of the two access
        // maps can overlap at all, WITHOUT the forward-iteration ordering
        // constraint (map_3). The ordering constraint must only be applied in
        // the combined multi-dimensional analysis, because a single dimension
        // may use a different variable (e.g. inner-loop indvar k) whose valid
        // values span across multiple outer-loop iterations.
        polyhedral::IslSet range_1(isl_map_range(map_1.release()));
        polyhedral::IslSet range_2(isl_map_range(map_2.release()));
        if (!range_1 || !range_2) {
            continue;
        }
        polyhedral::IslSet overlap(isl_set_intersect(range_1.release(), range_2.release()));
        bool disjoint = overlap ? isl_set_is_empty(overlap.get()) : false;
        if (disjoint) {
            return DependenceDeltas{true, "", {}};
        }
    }

    // Build combined analysis on all dimensions together
    auto maps = expressions_to_intersection_map_str(expr1_delinearized, expr2_delinearized, indvar, assums1, assums2);

    polyhedral::IslCtx ctx;
    if (!ctx) {
        return DependenceDeltas{false, "", {}};
    }

    polyhedral::IslMap map_1(isl_map_read_from_str(ctx.get(), std::get<0>(maps).c_str()));
    polyhedral::IslMap map_2(isl_map_read_from_str(ctx.get(), std::get<1>(maps).c_str()));
    polyhedral::IslMap map_3(isl_map_read_from_str(ctx.get(), std::get<2>(maps).c_str()));
    if (!map_1 || !map_2 || !map_3) {
        // Conservative: assume dependence exists when isl can't analyze
        return DependenceDeltas{false, "", {}};
    }

    // Compute alias pairs: { iter_1 -> iter_2 : access_1(iter_1) = access_2(iter_2) }
    // with monotonicity constraint from map_3.
    //
    // Compose access maps directly: apply_range(map_1, reverse(map_2)) gives
    // { iter_1 -> iter_2 : access_1(iter_1) = access_2(iter_2) }, then
    // intersect with reverse(map_3) for the monotonicity constraint.
    // This correctly captures cross-dimensional dependence vectors for both
    // square (n_iter == n_access) and rectangular (n_iter > n_access) cases.
    polyhedral::IslMap map_2_inv(isl_map_reverse(map_2.release()));
    if (!map_2_inv) {
        return DependenceDeltas{false, "", {}};
    }
    polyhedral::IslMap alias_unconstrained(isl_map_apply_range(map_1.release(), map_2_inv.release()));
    if (!alias_unconstrained) {
        return DependenceDeltas{false, "", {}};
    }
    polyhedral::IslMap mono(isl_map_reverse(map_3.release()));
    if (!mono) {
        return DependenceDeltas{false, "", {}};
    }
    polyhedral::IslMap alias_pairs(isl_map_intersect(alias_unconstrained.release(), mono.release()));
    if (!alias_pairs) {
        return DependenceDeltas{false, "", {}};
    }

    if (isl_map_is_empty(alias_pairs.get())) {
        return DependenceDeltas{true, "", {}};
    }

    // isl_map_deltas requires matching domain/range dimensions
    int n_in = isl_map_dim(alias_pairs.get(), isl_dim_in);
    int n_out = isl_map_dim(alias_pairs.get(), isl_dim_out);
    if (n_in != n_out) {
        // Dependence exists but can't compute deltas
        return DependenceDeltas{false, "", {}};
    }

    // Compute delta set: {d : exists i1,i2 in alias_pairs, d = i2 - i1}
    polyhedral::IslSet deltas(isl_map_deltas(alias_pairs.release()));
    if (!deltas || isl_set_is_empty(deltas.get())) {
        // Conservative: if deltas computation fails, assume dependence
        return DependenceDeltas{false, "", {}};
    }

    // Extract dimension names from the delta set
    int n_dims = isl_set_dim(deltas.get(), isl_dim_set);
    std::vector<std::string> dimensions;
    dimensions.reserve(n_dims);
    for (int i = 0; i < n_dims; i++) {
        const char* name = isl_set_get_dim_name(deltas.get(), isl_dim_set, i);
        if (name) {
            // Delta dim names are like "i_1" — strip the "_1" suffix
            std::string dim_name(name);
            if (dim_name.size() > 2 && dim_name.substr(dim_name.size() - 2) == "_1") {
                dim_name = dim_name.substr(0, dim_name.size() - 2);
            }
            dimensions.push_back(dim_name);
        } else {
            dimensions.push_back("d" + std::to_string(i));
        }
    }

    // Serialize to string
    char* str = isl_set_to_str(deltas.get());
    if (!str) {
        return DependenceDeltas{false, "", {}};
    }
    std::string deltas_str(str);
    free(str);

    DependenceDeltas result;
    result.empty = false;
    result.deltas_str = deltas_str;
    result.dimensions = dimensions;
    return result;
}

bool is_disjoint_monotonic(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
) {
    // TODO: Handle assumptions1 and assumptions2

    for (size_t i = 0; i < expr1.size(); i++) {
        auto& dim1 = expr1[i];
        if (expr2.size() <= i) {
            continue;
        }
        auto& dim2 = expr2[i];
        if (!symbolic::eq(dim1, dim2)) {
            continue;
        }

        // Collect all symbols
        symbolic::SymbolSet syms;
        for (auto& sym : symbolic::atoms(dim1)) {
            syms.insert(sym);
        }

        // Collect all non-constant symbols
        bool can_analyze = true;
        for (auto& sym : syms) {
            if (!assums1.at(sym).constant()) {
                if (sym->get_name() != indvar->get_name()) {
                    can_analyze = false;
                    break;
                }
            }
        }
        if (!can_analyze) {
            continue;
        }

        // Check if both dimensions are monotonic in non-constant symbols
        if (is_monotonic(dim1, indvar, assums1)) {
            return true;
        }
    }

    return false;
}

bool intersects(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
) {
    return !dependence_deltas(expr1, expr2, indvar, assums1, assums2).empty;
}

DependenceDeltas dependence_deltas(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    AssumptionsBounds& bounds1,
    AssumptionsBounds& bounds2
) {
    if (is_disjoint_monotonic(expr1, expr2, indvar, bounds1.assums(), bounds2.assums())) {
        return DependenceDeltas{true, "", {}};
    }
    return compute_deltas_isl(expr1, expr2, indvar, bounds1, bounds2);
}

DependenceDeltas dependence_deltas(
    const MultiExpression& expr1,
    const MultiExpression& expr2,
    const Symbol indvar,
    const Assumptions& assums1,
    const Assumptions& assums2
) {
    AssumptionsBounds b1(assums1);
    AssumptionsBounds b2(assums2);
    return dependence_deltas(expr1, expr2, indvar, b1, b2);
}

} // namespace maps
} // namespace symbolic
} // namespace sdfg
