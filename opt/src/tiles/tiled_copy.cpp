#include "sdfg/tiles/tiled_copy.h"

#include <algorithm>

#include "sdfg/symbolic/extreme_values.h"

namespace sdfg {
namespace tiles {

// Row-major delinearization of a flat index into per-dim coordinates (dim 0 slowest).
symbolic::MultiExpression delinearize_rowmajor(const symbolic::Expression& flat, const symbolic::MultiExpression& sizes) {
    symbolic::MultiExpression coords;
    symbolic::Expression remainder = flat;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i + 1 < sizes.size()) {
            symbolic::Expression divisor = symbolic::integer(1);
            for (size_t j = i + 1; j < sizes.size(); ++j) {
                divisor = symbolic::mul(divisor, sizes[j]);
            }
            coords.push_back(symbolic::div(remainder, divisor));
            remainder = symbolic::mod(remainder, divisor);
        } else {
            coords.push_back(remainder);
        }
    }
    return coords;
}

symbolic::Condition TileGuard::predicate(const symbolic::MultiExpression& coords) const {
    symbolic::Condition guard = SymEngine::boolTrue;
    for (const auto& d : dims) {
        auto global = symbolic::add(d.base, coords.at(d.axis));
        guard = symbolic::And(guard, symbolic::Le(global, d.max));
    }
    return guard;
}

symbolic::Condition TileGuard::predicate(const symbolic::Expression& flat) const {
    return predicate(delinearize_rowmajor(flat, tile_sizes));
}

void TileGuard::collect_symbols(symbolic::SymbolSet& set) const {
    for (const auto& s : tile_sizes) {
        for (const auto& a : symbolic::atoms(s)) {
            set.insert(a);
        }
    }
    for (const auto& d : dims) {
        for (const auto& a : symbolic::atoms(d.base)) {
            set.insert(a);
        }
        for (const auto& a : symbolic::atoms(d.max)) {
            set.insert(a);
        }
    }
}

void TileGuard::replace_symbols(const symbolic::ExpressionMapping& replacements) {
    for (auto& s : tile_sizes) {
        s = symbolic::subs(s, replacements);
    }
    for (auto& d : dims) {
        d.base = symbolic::subs(d.base, replacements);
        d.max = symbolic::subs(d.max, replacements);
    }
}

namespace {

// Provably `x >= y` for every symbol value: a non-negative constant difference, or for min/max
// operands the matching componentwise argument relation.
bool provably_ge(const symbolic::Expression& x, const symbolic::Expression& y) {
    auto diff = symbolic::simplify(symbolic::expand(symbolic::sub(x, y)));
    if (SymEngine::is_a<SymEngine::Integer>(*diff)) {
        return SymEngine::rcp_static_cast<const SymEngine::Integer>(diff)->as_int() >= 0;
    }
    // min(X) >= y if every X_i >= y; x >= max(Y) if x >= every Y_j.
    if (SymEngine::is_a<SymEngine::Min>(*x)) {
        for (const auto& a : x->get_args()) {
            if (!provably_ge(a, y)) {
                return false;
            }
        }
        return true;
    }
    if (SymEngine::is_a<SymEngine::Max>(*y)) {
        for (const auto& b : y->get_args()) {
            if (!provably_ge(x, b)) {
                return false;
            }
        }
        return true;
    }
    // x >= min(Y) if x >= some Y_j; max(X) >= y if some X_i >= y.
    if (SymEngine::is_a<SymEngine::Min>(*y)) {
        for (const auto& b : y->get_args()) {
            if (provably_ge(x, b)) {
                return true;
            }
        }
    }
    if (SymEngine::is_a<SymEngine::Max>(*x)) {
        for (const auto& a : x->get_args()) {
            if (provably_ge(a, y)) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

symbolic::Expression simplify_tile_bound(
    const symbolic::Expression& max, const symbolic::Expression& base, const symbolic::Expression& size
) {
    if (max.is_null()) {
        return max;
    }
    const bool is_min = SymEngine::is_a<SymEngine::Min>(*max);
    if (!is_min && !SymEngine::is_a<SymEngine::Max>(*max)) {
        return max;
    }
    // The coordinate never exceeds base + size - 1, so a min argument at or above it never binds.
    symbolic::Expression tile_last = size.is_null() ? symbolic::Expression(SymEngine::null)
                                                    : symbolic::add(base, symbolic::sub(size, symbolic::one()));
    std::vector<symbolic::Expression> args;
    for (const auto& a : max->get_args()) {
        args.push_back(simplify_tile_bound(a, base, size));
    }
    std::vector<symbolic::Expression> kept;
    for (size_t i = 0; i < args.size(); ++i) {
        if (is_min && !tile_last.is_null() && provably_ge(args[i], tile_last)) {
            continue;
        }
        // Drop an argument dominated by another (for min: one that is >= another; for max: <=).
        bool dominated = false;
        for (size_t j = 0; j < args.size() && !dominated; ++j) {
            if (i == j) {
                continue;
            }
            bool ge = is_min ? provably_ge(args[i], args[j]) : provably_ge(args[j], args[i]);
            // Mutual dominance (equal arguments): keep the first.
            bool le = is_min ? provably_ge(args[j], args[i]) : provably_ge(args[i], args[j]);
            dominated = ge && (!le || j < i);
        }
        if (!dominated) {
            kept.push_back(args[i]);
        }
    }
    if (kept.empty()) {
        // Every min argument is above the tile: the bound never binds within it.
        return tile_last.is_null() ? max : tile_last;
    }
    if (kept.size() == 1) {
        return kept.front();
    }
    SymEngine::vec_basic v(kept.begin(), kept.end());
    return is_min ? SymEngine::min(v) : SymEngine::max(v);
}

bool TileGuard::simplify() {
    bool changed = false;
    for (auto& d : dims) {
        auto size = d.axis < tile_sizes.size() ? tile_sizes.at(d.axis) : symbolic::Expression(SymEngine::null);
        auto s = simplify_tile_bound(d.max, d.base, size);
        if (!symbolic::eq(s, d.max)) {
            d.max = s;
            changed = true;
        }
    }
    return changed;
}

bool TileGuard::discharge(const symbolic::SymbolSet& parameters, const symbolic::Assumptions& assumptions) {
    bool simplified = simplify();
    if (assumptions.empty()) {
        return simplified;
    }
    std::vector<Dim> kept;
    for (const auto& d : dims) {
        // Worst case over the sweep: the coordinate at its maximum (size - 1). If
        // `base + (size - 1) <= max` is provable, the dim never overshoots.
        auto size = d.axis < tile_sizes.size() ? tile_sizes.at(d.axis) : symbolic::Expression(SymEngine::null);
        symbolic::Expression probe = size.is_null() ? d.base
                                                    : symbolic::add(d.base, symbolic::sub(size, symbolic::integer(1)));
        if (symbolic::is_le(probe, d.max, parameters, assumptions, /*tight=*/true)) {
            continue;
        }
        kept.push_back(d);
    }
    if (kept.size() == dims.size()) {
        return simplified;
    }
    dims = std::move(kept);
    return true;
}

bool coalesce_sweep_order(TiledCopy& plan, TileGuard& guard) {
    const auto& strides = plan.src.strides();
    const size_t n = strides.size();
    size_t unit = n;
    for (size_t d = 0; d < n; ++d) {
        if (symbolic::eq(strides[d], symbolic::one())) {
            unit = d;
            break;
        }
    }
    if (unit + 1 >= n) {
        return false;
    }
    std::vector<size_t> order;
    for (size_t d = 0; d < n; ++d) {
        if (d != unit) {
            order.push_back(d);
        }
    }
    order.push_back(unit);
    auto permute = [&](const symbolic::MultiExpression& v) {
        symbolic::MultiExpression out;
        for (size_t d : order) {
            out.push_back(v.at(d));
        }
        return out;
    };
    plan.src = Layout(permute(plan.src.shape()), permute(plan.src.strides()), plan.src.offset());
    plan.dst = Layout(permute(plan.dst.shape()), permute(plan.dst.strides()), plan.dst.offset());
    if (guard.tile_sizes.size() == n) {
        guard.tile_sizes = permute(guard.tile_sizes);
    }
    for (auto& d : guard.dims) {
        d.axis = static_cast<size_t>(std::find(order.begin(), order.end(), d.axis) - order.begin());
    }
    return true;
}

} // namespace tiles
} // namespace sdfg
