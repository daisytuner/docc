#pragma once

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/copy_node.h"

namespace sdfg {
namespace fp_math {

/// Whether the k x n B of the MatMul or GEMM @p node is stored transposed, its rows along its columns' elements.
bool b_stored_transposed(const data_flow::LibraryNode& node);

/// The copy read_b_row_major() makes of B, and the contraction that reads it.
struct RowMajorB {
    math::tensor::TensorCopyNode& copy;
    data_flow::LibraryNode& contraction;
};

/**
 * Lets the MatMul or GEMM @p node, whose k x n B is stored transposed, read a row-major copy of B made by a
 * TensorCopy node in a block before its own; each matrix of a batch is transposed on its own. The contraction is a new
 * node for a GEMM, which replaces @p node.
 *
 * Scales of B's columns, as the direct rescaling computes them, then reduce each column along the copy's rows, and
 * the slices or quantized B are row-major as well.
 */
RowMajorB read_b_row_major(builder::StructuredSDFGBuilder& builder, data_flow::LibraryNode& node);

} // namespace fp_math
} // namespace sdfg
