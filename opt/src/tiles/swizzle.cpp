#include "sdfg/tiles/swizzle.h"

#include <cstdlib>

namespace sdfg {
namespace tiles {

symbolic::Expression Swizzle::apply(const symbolic::Expression& offset) const {
    if (is_identity()) {
        return offset;
    }
    // yyy = (offset >> (base + |shift|)) & ((1<<bits)-1);  offset ^ (yyy << base)
    const int dist = base + std::abs(shift);
    const symbolic::Expression two_dist = symbolic::pow(symbolic::integer(2), symbolic::integer(dist));
    const symbolic::Expression two_bits = symbolic::pow(symbolic::integer(2), symbolic::integer(bits));
    const symbolic::Expression two_base = symbolic::pow(symbolic::integer(2), symbolic::integer(base));
    const symbolic::Expression field = symbolic::mod(symbolic::div(offset, two_dist), two_bits);
    return symbolic::bit_xor(offset, symbolic::mul(field, two_base));
}

symbolic::Expression ComposedLayout::apply(const symbolic::Expression& index) const {
    return swizzle.apply(layout.apply(index));
}

symbolic::Expression ComposedLayout::apply_coords(const symbolic::MultiExpression& coords) const {
    return swizzle.apply(layout.apply_coords(coords));
}

} // namespace tiles
} // namespace sdfg
