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

// Tile dims (or indices) in storage order: reversed for the column-major placement.
symbolic::MultiExpression storage_order(const symbolic::MultiExpression& v, Placement placement) {
    if (placement == Placement::ColMajor) {
        return symbolic::MultiExpression(v.rbegin(), v.rend());
    }
    return v;
}

// Whether each per-slot block is stored as one flat axis.
bool flat_block(ConflictPolicy conflict) {
    return conflict == ConflictPolicy::PadSlot || conflict == ConflictPolicy::Swizzle;
}

// The nested tile dims as allocated: storage order, the innermost row widened by a row pad.
symbolic::MultiExpression stored_tile(const PackedBuffer& buffer) {
    auto out = storage_order(buffer.tile_sizes, buffer.placement);
    if (buffer.conflict == ConflictPolicy::PadRow && buffer.pad > 0 && !out.empty()) {
        out.back() = symbolic::add(out.back(), symbolic::integer(static_cast<int64_t>(buffer.pad)));
    }
    return out;
}

} // namespace

ComposedLayout PackedBuffer::layout() const {
    symbolic::MultiExpression shape, stride;
    if (placement == Placement::Flat) {
        symbolic::MultiExpression all = slot_sizes;
        all.insert(all.end(), tile_sizes.begin(), tile_sizes.end());
        for (size_t i = 0; i < all.size(); ++i) {
            shape.push_back(all[i]);
            stride.push_back(product_of(all, i + 1, all.size()));
        }
        return ComposedLayout{Swizzle{}, Layout(shape, stride, symbolic::integer(0))};
    }

    // Slot dims (outermost): each owns a contiguous per-slot block.
    const auto block = slot_stride();
    for (size_t i = 0; i < slot_sizes.size(); ++i) {
        shape.push_back(slot_sizes[i]);
        stride.push_back(symbolic::mul(block, product_of(slot_sizes, i + 1, slot_sizes.size())));
    }
    // Tile dims: strides of the stored dims (a flat block keeps the natural, unpadded order).
    const auto stored = flat_block(conflict) ? storage_order(tile_sizes, placement) : stored_tile(*this);
    const size_t n = tile_sizes.size();
    for (size_t j = 0; j < n; ++j) {
        const size_t s = placement == Placement::ColMajor ? n - 1 - j : j;
        shape.push_back(tile_sizes[j]);
        stride.push_back(product_of(stored, s + 1, n));
    }
    Layout base(shape, stride, symbolic::integer(0));

    // Swizzle: XOR the low log2(block) bits (the tile index) by the slot index,
    // equivalent to bit_xor(tile_linearize, slot_linearize) when the block is a
    // constant power of two (and the slot fits in it).
    Swizzle sw;
    const auto tile_total = tile_total_size();
    if (conflict == ConflictPolicy::Swizzle && SymEngine::is_a<SymEngine::Integer>(*tile_total)) {
        long total = SymEngine::rcp_static_cast<const SymEngine::Integer>(tile_total)->as_int();
        if (total > 1 && (total & (total - 1)) == 0) {
            int bits = 0;
            while ((1L << bits) < total) {
                ++bits;
            }
            sw = Swizzle{bits, 0, bits};
        }
    }
    return ComposedLayout{sw, base};
}

symbolic::Expression PackedBuffer::total_size() const {
    return symbolic::mul(product_of(slot_sizes, 0, slot_sizes.size()), slot_stride());
}

symbolic::Expression PackedBuffer::tile_total_size() const {
    return product_of(tile_sizes, 0, tile_sizes.size());
}

symbolic::Expression PackedBuffer::slot_stride() const {
    if (placement == Placement::Flat || conflict == ConflictPolicy::Swizzle) {
        return tile_total_size();
    }
    if (conflict == ConflictPolicy::PadSlot) {
        return symbolic::add(tile_total_size(), symbolic::integer(static_cast<int64_t>(pad)));
    }
    const auto stored = stored_tile(*this);
    return product_of(stored, 0, stored.size());
}

symbolic::MultiExpression PackedBuffer::axes() const {
    if (placement == Placement::Flat) {
        return {total_size()};
    }
    symbolic::MultiExpression out = slot_sizes;
    if (flat_block(conflict)) {
        out.push_back(slot_stride());
    } else {
        const auto stored = stored_tile(*this);
        out.insert(out.end(), stored.begin(), stored.end());
    }
    return out;
}

symbolic::MultiExpression PackedBuffer::
    subset(const symbolic::MultiExpression& slot_indices, const symbolic::MultiExpression& tile_indices) const {
    if (placement == Placement::Flat) {
        symbolic::MultiExpression sizes = slot_sizes, indices = slot_indices;
        sizes.insert(sizes.end(), tile_sizes.begin(), tile_sizes.end());
        indices.insert(indices.end(), tile_indices.begin(), tile_indices.end());
        return {rowmajor(sizes, indices)};
    }
    symbolic::MultiExpression out = slot_indices;
    const auto indices = storage_order(tile_indices, placement);
    if (flat_block(conflict)) {
        auto inner = rowmajor(storage_order(tile_sizes, placement), indices);
        if (conflict == ConflictPolicy::Swizzle) {
            inner = symbolic::bit_xor(inner, rowmajor(slot_sizes, slot_indices));
        }
        out.push_back(inner);
    } else {
        out.insert(out.end(), indices.begin(), indices.end());
    }
    // A degenerate (all extent-1, no-slot) buffer is a single [1] element at index 0.
    if (out.empty()) {
        out.push_back(symbolic::integer(0));
    }
    return out;
}

} // namespace tiles
} // namespace sdfg
