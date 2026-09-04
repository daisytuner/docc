#include "sdfg/tiles/buffer_layout.h"

namespace sdfg {
namespace tiles {

namespace {

// Product of `v[begin:end)` (empty range = 1).
symbolic::Expression product_of(const symbolic::MultiExpression& v, size_t begin, size_t end) {
    symbolic::Expression p = symbolic::integer(1);
    for (size_t i = begin; i < end; ++i) {
        p = symbolic::mul(p, v[i]);
    }
    return p;
}

// Row-major linear index of `indices` over `sizes`.
symbolic::Expression rowmajor(const symbolic::MultiExpression& sizes, const symbolic::MultiExpression& indices) {
    symbolic::Expression linear = symbolic::integer(0);
    symbolic::Expression stride = symbolic::integer(1);
    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
        linear = symbolic::add(linear, symbolic::mul(indices[i], stride));
        if (i < static_cast<int>(sizes.size())) {
            stride = symbolic::mul(stride, sizes[i]);
        }
    }
    return linear;
}

} // namespace

ComposedLayout buffer_layout(
    const symbolic::MultiExpression& slot_sizes,
    const symbolic::MultiExpression& tile_sizes,
    BufferKind kind,
    const symbolic::Expression& inner_stride
) {
    symbolic::Expression tile_total = product_of(tile_sizes, 0, tile_sizes.size());
    // Padded widens the per-slot block; every other kind packs the natural block.
    symbolic::Expression per_slot_block = (kind == BufferKind::Padded) ? inner_stride : tile_total;

    symbolic::MultiExpression shape, stride;
    // Slot dims (outermost): each owns a contiguous per-slot block.
    for (size_t i = 0; i < slot_sizes.size(); ++i) {
        shape.push_back(slot_sizes[i]);
        stride.push_back(symbolic::mul(per_slot_block, product_of(slot_sizes, i + 1, slot_sizes.size())));
    }
    // Tile dims (innermost): row-major within one per-slot block.
    for (size_t j = 0; j < tile_sizes.size(); ++j) {
        shape.push_back(tile_sizes[j]);
        stride.push_back(product_of(tile_sizes, j + 1, tile_sizes.size()));
    }
    Layout base(shape, stride, symbolic::integer(0));

    // Swizzle: XOR the low log2(block) bits (the tile index) by the slot index,
    // equivalent to bit_xor(tile_linearize, slot_linearize) when the block is a
    // constant power of two (and the slot fits in it).
    Swizzle sw;
    if (kind == BufferKind::Swizzle && SymEngine::is_a<SymEngine::Integer>(*tile_total)) {
        long n = SymEngine::rcp_static_cast<const SymEngine::Integer>(tile_total)->as_int();
        if (n > 1 && (n & (n - 1)) == 0) {
            int bits = 0;
            while ((1L << bits) < n) {
                ++bits;
            }
            sw = Swizzle{bits, 0, bits};
        }
    }
    return ComposedLayout{sw, base};
}

symbolic::Expression PackedBuffer::total_size() const {
    return symbolic::mul(product_of(slot_sizes, 0, slot_sizes.size()), product_of(tile_sizes, 0, tile_sizes.size()));
}

symbolic::Expression PackedBuffer::tile_total_size() const { return product_of(tile_sizes, 0, tile_sizes.size()); }

symbolic::Expression PackedBuffer::inner_stride() const {
    auto total = tile_total_size();
    // Pad only a constant block; a symbolic block cannot be safely padded.
    if (!SymEngine::is_a<SymEngine::Integer>(*total)) {
        return total;
    }
    auto n = SymEngine::rcp_static_cast<const SymEngine::Integer>(total)->as_int();
    // Cooperative-store conflict avoidance: pad the per-slot stride congruent to the
    // coop warp span (mod 32), rounded up to a multiple of 4 so a whole shared row is
    // float4-addressable; else fall back to the next odd (coprime-with-32) stride.
    if (coop_warp_span > 0) {
        long target = static_cast<long>(coop_warp_span % 32);
        long add = ((target - (n % 32)) % 32 + 32) % 32;
        long padded = (n + add + 3) & ~3L;
        return symbolic::integer(padded);
    }
    if (n % 2 == 0) {
        return symbolic::integer(n + 1);
    }
    return total;
}

std::vector<symbolic::Expression> PackedBuffer::delinearize_tile(const symbolic::Expression& flat) const {
    std::vector<symbolic::Expression> decomp;
    symbolic::Expression remainder = flat;
    for (size_t i = 0; i < tile_sizes.size(); ++i) {
        if (i + 1 < tile_sizes.size()) {
            auto divisor = product_of(tile_sizes, i + 1, tile_sizes.size());
            decomp.push_back(symbolic::div(remainder, divisor));
            remainder = symbolic::mod(remainder, divisor);
        } else {
            decomp.push_back(remainder);
        }
    }
    return decomp;
}

symbolic::MultiExpression PackedBuffer::axes() const {
    symbolic::MultiExpression out = slot_sizes;
    symbolic::Expression tile_total = product_of(tile_sizes, 0, tile_sizes.size());
    switch (kind) {
        case BufferKind::MultiDim:
            out.insert(out.end(), tile_sizes.begin(), tile_sizes.end());
            break;
        case BufferKind::Padded:
            out.push_back(inner_stride()); // one flat, padded per-slot block
            break;
        case BufferKind::Swizzle:
            out.push_back(tile_total); // one flat, natural per-slot block
            break;
        case BufferKind::Linearized:
            return {symbolic::mul(product_of(slot_sizes, 0, slot_sizes.size()), tile_total)};
    }
    return out;
}

symbolic::MultiExpression PackedBuffer::
    subset(const symbolic::MultiExpression& slot_indices, const symbolic::MultiExpression& tile_indices) const {
    symbolic::MultiExpression out = slot_indices;
    switch (kind) {
        case BufferKind::MultiDim:
            out.insert(out.end(), tile_indices.begin(), tile_indices.end());
            break;
        case BufferKind::Padded:
            out.push_back(rowmajor(tile_sizes, tile_indices));
            break;
        case BufferKind::Swizzle:
            out.push_back(symbolic::bit_xor(rowmajor(tile_sizes, tile_indices), rowmajor(slot_sizes, slot_indices)));
            break;
        case BufferKind::Linearized: {
            symbolic::MultiExpression sizes = slot_sizes, indices = slot_indices;
            sizes.insert(sizes.end(), tile_sizes.begin(), tile_sizes.end());
            indices.insert(indices.end(), tile_indices.begin(), tile_indices.end());
            return {rowmajor(sizes, indices)};
        }
    }
    // A degenerate (all extent-1, no-slot) buffer is a single [1] element at index 0.
    if (out.empty()) {
        out.push_back(symbolic::integer(0));
    }
    return out;
}

} // namespace tiles
} // namespace sdfg
