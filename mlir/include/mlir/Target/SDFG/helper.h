#pragma once

#include <cstddef>
#include <unordered_map>
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Types.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/symbolic/symbolic.h"

namespace mlir {
namespace sdfg {

bool is_sdfg_primitive(Type type);
bool is_vector_of_sdfg_primitive(Type type);
bool is_tensor_of_sdfg_primitive(Type type);
bool is_vector_or_tensor_of_sdfg_primitive(Type type);

::sdfg::symbolic::Expression affine_expr_to_symbolic_expr(
    AffineExpr affine_expr,
    const std::unordered_map<size_t, ::sdfg::symbolic::Symbol>& dimensions = {},
    const std::unordered_map<size_t, ::sdfg::symbolic::Symbol>& symbols = {},
    bool strict = false
);
::sdfg::data_flow::Subset affine_map_to_subset(
    AffineMap affine_map, const ::sdfg::symbolic::SymbolVec& dimensions, const ::sdfg::symbolic::SymbolVec& symbols = {}
);

} // namespace sdfg
} // namespace mlir
