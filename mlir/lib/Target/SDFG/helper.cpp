#include "mlir/Target/SDFG/helper.h"

#include <cstddef>
#include <llvm/Support/Casting.h>
#include <stdexcept>
#include <unordered_map>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/symbolic/symbolic.h"

namespace mlir {
namespace sdfg {

bool is_sdfg_primitive(Type type) {
    if (auto int_type = llvm::dyn_cast_or_null<IntegerType>(type)) {
        switch (int_type.getWidth()) {
            case 1:
            case 8:
            case 16:
            case 32:
            case 64:
            case 128:
                return true;
            default:
                return false;
        }
    }
    return type.isF16() || type.isBF16() || type.isF32() || type.isF64() || type.isF80() || type.isF128();
}

bool is_vector_of_sdfg_primitive(Type type) {
    if (auto vector_type = llvm::dyn_cast_or_null<VectorType>(type)) {
        return is_sdfg_primitive(vector_type.getElementType());
    }
    return false;
}

bool is_tensor_of_sdfg_primitive(Type type) {
    if (auto tensor_type = llvm::dyn_cast_or_null<TensorType>(type)) {
        return is_sdfg_primitive(tensor_type.getElementType());
    }
    return false;
}

bool is_vector_or_tensor_of_sdfg_primitive(Type type) {
    return is_vector_of_sdfg_primitive(type) || is_tensor_of_sdfg_primitive(type);
}

::sdfg::symbolic::Expression affine_expr_to_symbolic_expr(
    AffineExpr affine_expr,
    const std::unordered_map<size_t, ::sdfg::symbolic::Symbol>& dimensions,
    const std::unordered_map<size_t, ::sdfg::symbolic::Symbol>& symbols,
    bool strict
) {
    switch (affine_expr.getKind()) {
        case AffineExprKind::Add: {
            auto affine_binary_op_expr = llvm::dyn_cast<AffineBinaryOpExpr>(affine_expr);
            auto lhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getLHS(), dimensions, symbols, strict);
            auto rhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getRHS(), dimensions, symbols, strict);
            return ::sdfg::symbolic::add(lhs, rhs);
        }
        case AffineExprKind::Mul: {
            auto affine_binary_op_expr = llvm::dyn_cast<AffineBinaryOpExpr>(affine_expr);
            auto lhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getLHS(), dimensions, symbols, strict);
            auto rhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getRHS(), dimensions, symbols, strict);
            return ::sdfg::symbolic::mul(lhs, rhs);
        }
        case AffineExprKind::Mod: {
            auto affine_binary_op_expr = llvm::dyn_cast<AffineBinaryOpExpr>(affine_expr);
            auto lhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getLHS(), dimensions, symbols, strict);
            auto rhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getRHS(), dimensions, symbols, strict);
            return ::sdfg::symbolic::mod(lhs, rhs);
        }
        case AffineExprKind::FloorDiv: {
            auto affine_binary_op_expr = llvm::dyn_cast<AffineBinaryOpExpr>(affine_expr);
            auto lhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getLHS(), dimensions, symbols, strict);
            auto rhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getRHS(), dimensions, symbols, strict);
            return ::sdfg::symbolic::div(lhs, rhs);
        }
        case AffineExprKind::CeilDiv: {
            auto affine_binary_op_expr = llvm::dyn_cast<AffineBinaryOpExpr>(affine_expr);
            auto lhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getLHS(), dimensions, symbols, strict);
            auto rhs = affine_expr_to_symbolic_expr(affine_binary_op_expr.getRHS(), dimensions, symbols, strict);
            return ::sdfg::symbolic::divide_ceil(lhs, rhs);
        }
        case AffineExprKind::Constant: {
            auto affine_constant_expr = llvm::dyn_cast<AffineConstantExpr>(affine_expr);
            return ::sdfg::symbolic::integer(affine_constant_expr.getValue());
        }
        case AffineExprKind::DimId: {
            auto affine_dim_expr = llvm::dyn_cast<AffineDimExpr>(affine_expr);
            size_t position = affine_dim_expr.getPosition();
            if (dimensions.contains(position)) {
                return dimensions.at(position);
            } else {
                if (strict) {
                    throw std::runtime_error(
                        "affine_expr_to_symbolic_expr: Cannot convert dimension at position " + std::to_string(position)
                    );
                }
                return ::sdfg::symbolic::symbol("d" + std::to_string(position));
            }
        }
        case AffineExprKind::SymbolId: {
            auto affine_symbol_expr = llvm::cast<AffineSymbolExpr>(affine_expr);
            size_t position = affine_symbol_expr.getPosition();
            if (symbols.contains(position)) {
                return symbols.at(position);
            } else {
                if (strict) {
                    throw std::runtime_error(
                        "affine_expr_to_symbolic_expr: Cannot convert symbol at position " + std::to_string(position)
                    );
                }
                return ::sdfg::symbolic::symbol("s" + std::to_string(position));
            }
        }
    }
}

::sdfg::data_flow::Subset affine_map_to_subset(
    AffineMap affine_map, const ::sdfg::symbolic::SymbolVec& dimensions, const ::sdfg::symbolic::SymbolVec& symbols
) {
    if (affine_map.getNumDims() != dimensions.size()) {
        throw std::runtime_error(
            "affine_map_to_subset: The number of affine map dimensions and the number of given dimensions does not "
            "match"
        );
    }
    if (affine_map.getNumSymbols() != symbols.size()) {
        throw std::runtime_error(
            "affine_map_to_subset: The number of affine map dimensions and the number of given dimensions does not "
            "match"
        );
    }

    // Construct maps
    std::unordered_map<size_t, ::sdfg::symbolic::Symbol> dimensions_map, symbols_map;
    for (size_t i = 0; i < dimensions.size(); i++) {
        dimensions_map.insert({i, dimensions.at(i)});
    }
    for (size_t i = 0; i < symbols.size(); i++) {
        symbols_map.insert({i, symbols.at(i)});
    }

    // Fill subset
    ::sdfg::data_flow::Subset subset;
    subset.reserve(affine_map.getNumResults());
    for (auto affine_expr : affine_map.getResults()) {
        subset.push_back(affine_expr_to_symbolic_expr(affine_expr, dimensions_map, symbols_map, true));
    }

    return subset;
}

} // namespace sdfg
} // namespace mlir
