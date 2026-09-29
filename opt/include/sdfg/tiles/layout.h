#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"

namespace sdfg {
namespace tiles {

/// The tiles geometry type is `math::tensor::TensorLayout` — the single source of
/// truth for a `(shape, strides, offset)` affine element map. A tile coordinate
/// `x` addresses `offset + sum_k x_k * strides_k` via `TensorLayout::resolve_element`.
using Layout = math::tensor::TensorLayout;

} // namespace tiles
} // namespace sdfg
