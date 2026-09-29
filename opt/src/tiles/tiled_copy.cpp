#include "sdfg/tiles/tiled_copy.h"

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

bool TileGuard::discharge(const symbolic::SymbolSet& parameters, const symbolic::Assumptions& assumptions) {
    if (assumptions.empty()) {
        return false;
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
        return false;
    }
    dims = std::move(kept);
    return true;
}

} // namespace tiles
} // namespace sdfg
