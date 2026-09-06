#include "sdfg/tiles/tiled_copy.h"

#include <string>

namespace sdfg {
namespace tiles {

namespace {

bool is_int(const symbolic::Expression& e) { return SymEngine::is_a<SymEngine::Integer>(*e); }
long long as_ll(const symbolic::Expression& e) {
    return SymEngine::rcp_static_cast<const SymEngine::Integer>(e)->as_int();
}

bool prov_eq(const symbolic::Expression& a, const symbolic::Expression& b) {
    if (is_int(a) && is_int(b)) {
        return as_ll(a) == as_ll(b);
    }
    return symbolic::eq(symbolic::simplify(a), symbolic::simplify(b));
}

/// Integer size of a layout, or -1 when not a compile-time constant.
long long size_int(const Layout& l) {
    auto s = l.size();
    return is_int(s) ? as_ll(s) : -1;
}

/// True when the value run is unit-stride when viewed through @p over (so the
/// per-lane footprint is a single contiguous chunk in that space).
bool contiguous_over(const Layout& over, const Layout& value) {
    const long long v = size_int(value);
    if (v <= 1) {
        return true;
    }
    auto step =
        symbolic::sub(over.apply(value.apply(symbolic::integer(1))), over.apply(value.apply(symbolic::integer(0))));
    return prov_eq(step, symbolic::integer(1));
}

} // namespace

Layout TiledCopy::partition() const { return concat(value, thread); }

bool is_lane_contiguous(const Layout& dst, const Layout& thread) {
    if (size_int(thread) < 2) {
        return true; // a 0/1-lane partition is contiguous by default
    }
    auto step =
        symbolic::sub(dst.apply(thread.apply(symbolic::integer(1))), dst.apply(thread.apply(symbolic::integer(0))));
    return prov_eq(step, symbolic::integer(1));
}

std::optional<std::string> TiledCopy::verify(size_t elem_bytes) const {
    if (!prov_eq(src.size(), dst.size())) {
        return "src/dst tile size mismatch";
    }
    Layout part = partition();
    if (!prov_eq(part.size(), src.size())) {
        return "partition size != tile size";
    }
    if (!part.is_bijective()) {
        return "partition is not a bijection onto the tile (elements skipped or double-copied)";
    }
    if (!dst.is_injective()) {
        return "destination layout double-writes a buffer slot";
    }

    const long long width = size_int(value) * static_cast<long long>(elem_bytes);
    switch (atom) {
        case CopyAtom::ScalarSync:
            break;
        case CopyAtom::VectorSync:
        case CopyAtom::CpAsync:
            if (!contiguous_over(src, value) || !contiguous_over(dst, value)) {
                return "vector/async value run is not contiguous in src and dst";
            }
            if (width != 4 && width != 8 && width != 16) {
                return "transfer width " + std::to_string(width) + " not in {4,8,16}";
            }
            break;
        case CopyAtom::LaneContiguousDMA: {
            if (width != 4) {
                return "lane-contiguous DMA requires a 4-byte per-lane transfer";
            }
            // Consecutive lanes must land on consecutive dst slots (the
            // global_load_lds constraint).
            if (!is_lane_contiguous(dst, thread)) {
                return "lane-contiguous DMA: consecutive lanes are not consecutive in dst";
            }
            break;
        }
    }
    return std::nullopt;
}

} // namespace tiles
} // namespace sdfg
