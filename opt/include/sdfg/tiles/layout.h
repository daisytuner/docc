#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace tiles {

/**
 * @brief A small symbolic layout algebra for schedule-aware tiles.
 *
 * A @ref Layout maps a coordinate tuple to a linear element offset:
 * `L(x) = offset + sum_k x_k * stride_k` over `0 <= x_k < shape_k`
 * (colexicographic: dim 0 fastest).
 */
class Layout {
private:
    symbolic::MultiExpression shape_; ///< per-dim extents, first dim fastest
    symbolic::MultiExpression stride_; ///< per-dim strides, aligned with shape_
    symbolic::Expression offset_; ///< base offset in elements

public:
    Layout(
        symbolic::MultiExpression shape = {},
        symbolic::MultiExpression stride = {},
        symbolic::Expression offset = symbolic::integer(0)
    );

    /// The identity layout on `[0, n)`: shape {n}, stride {1}, offset 0.
    static Layout identity(const symbolic::Expression& n);

    /// Lift a geometry layout (from MemoryLayoutAnalysis) into the algebra.
    static Layout from_tensor(const math::tensor::TensorLayout& t);

    /// Project back to a geometry layout.
    math::tensor::TensorLayout to_tensor() const;

    const symbolic::MultiExpression& shape() const { return shape_; }

    const symbolic::MultiExpression& stride() const { return stride_; }

    const symbolic::Expression& offset() const { return offset_; }

    size_t rank() const { return shape_.size(); }

    /// `size = prod(shape)` — the domain cardinality.
    symbolic::Expression size() const;

    /// `cosize = offset + sum_k (s_k - 1)*stride_k + 1` — the span it touches.
    symbolic::Expression cosize() const;

    /// Colex-decompose a flat index into per-dim coordinates.
    symbolic::MultiExpression coords(const symbolic::Expression& index) const;

    /// `L(index)` = the linear element offset.
    symbolic::Expression apply(const symbolic::Expression& index) const;

    /// `L` at explicit per-dim coordinates (one per flat dim).
    symbolic::Expression apply_coords(const symbolic::MultiExpression& coords) const;

    /// Provably a permutation of `[0, size)` (dense, no gaps).
    bool is_bijective() const;

    /// Provably injective (no coordinate collides; gaps allowed).
    bool is_injective() const;

    bool operator==(const Layout& other) const;

    void collect_symbols(symbolic::SymbolSet& set) const;

    void replace_symbols(const symbolic::ExpressionMapping& replacements);

    std::string to_string() const;
};

std::ostream& operator<<(std::ostream& stream, const Layout& layout);

/// `A ++ B`: append `B`'s dims at their absolute strides. Assembles a
/// `(value, thread)` partition as `partition = concat(value, thread)`.
Layout concat(const Layout& A, const Layout& B);

/// Normal form: merge contiguous dims, drop size-1 dims. Idempotent and
/// function-preserving (`coalesce(A)(i) == A(i)`).
Layout coalesce(const Layout& A);

} // namespace tiles
} // namespace sdfg
