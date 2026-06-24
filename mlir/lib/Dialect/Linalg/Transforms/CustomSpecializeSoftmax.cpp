#include "mlir/Dialect/Linalg/Transforms/CustomSpecializeSoftmax.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace linalg {

bool LinalgGenericToLinalgSoftmax::checkIndexingMap(Attribute attr, long long dim) const {
    auto affine_map_attr = llvm::dyn_cast_or_null<AffineMapAttr>(attr);
    if (!affine_map_attr) {
        return false;
    }
    AffineMap affine_map = affine_map_attr.getAffineMap();
    if (affine_map.getNumDims() != affine_map.getNumResults()) {
        return false;
    }
    ArrayRef<AffineExpr> results = affine_map.getResults();
    for (long long i = 0, numDims = affine_map.getNumDims(); i < numDims; ++i) {
        if (i == dim) {
            auto expr = llvm::dyn_cast<AffineConstantExpr>(results[i]);
            if (!expr || expr.getValue() != 0) {
                return false;
            }
        } else {
            auto expr = llvm::dyn_cast<AffineDimExpr>(results[i]);
            if (!expr || expr.getPosition() != i) {
                return false;
            }
        }
    }
    return true;
}

bool LinalgGenericToLinalgSoftmax::checkIndexingMapCollapsed(Attribute attr, long long dim) const {
    auto affine_map_attr = llvm::dyn_cast_or_null<AffineMapAttr>(attr);
    if (!affine_map_attr) {
        return false;
    }
    AffineMap affine_map = affine_map_attr.getAffineMap();
    if (affine_map.getNumDims() != affine_map.getNumResults() + 1) {
        return false;
    }
    ArrayRef<AffineExpr> results = affine_map.getResults();
    for (long long i = 0, numDims = affine_map.getNumDims(); i < numDims; ++i) {
        if (i == dim) {
            continue;
        }
        long long offset = (i > dim) ? -1 : 0;
        auto expr = llvm::dyn_cast<AffineDimExpr>(results[i + offset]);
        if (!expr || expr.getPosition() != i) {
            return false;
        }
    }
    return true;
}

bool LinalgGenericToLinalgSoftmax::checkIndexingMaps(ArrayAttr maps, const std::vector<bool>& reduced_mask, long long dim)
    const {
    size_t rank = maps.size();
    if (rank != reduced_mask.size()) {
        return false;
    }
    for (size_t i = 0; i < rank; i++) {
        if (reduced_mask[i]) {
            if (!this->checkIndexingMap(maps[i], dim)) {
                return false;
            }
        } else {
            if (!this->checkIndexingMap(maps[i])) {
                return false;
            }
        }
    }
    return true;
}

bool LinalgGenericToLinalgSoftmax::checkIteratorTypes(ArrayAttr types, size_t rank, long long dim) const {
    if (rank != types.size()) {
        return false;
    }
    for (size_t i = 0; i < rank; i++) {
        auto iterator_type_attr = llvm::dyn_cast_or_null<IteratorTypeAttr>(types[i]);
        if (!iterator_type_attr) {
            return false;
        }
        if (static_cast<long long>(i) == dim) {
            if (iterator_type_attr.getValue() != utils::IteratorType::reduction) {
                return false;
            }
        } else {
            if (iterator_type_attr.getValue() != utils::IteratorType::parallel) {
                return false;
            }
        }
    }
    return true;
}

bool LinalgGenericToLinalgSoftmax::equalShapes(const ArrayRef<int64_t>& shape1, const ArrayRef<int64_t>& shape2) const {
    size_t rank = shape1.size();
    if (rank != shape2.size()) {
        return false;
    }
    for (size_t i = 0; i < rank; i++) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }
    return true;
}

long long LinalgGenericToLinalgSoftmax::
    determineDimension(const ArrayRef<int64_t>& full_shape, const ArrayRef<int64_t>& reduced_shape) const {
    long long rank = full_shape.size();
    if (rank != static_cast<long long>(reduced_shape.size())) {
        return -1;
    }
    long long dim = -1;
    for (long long i = 0; i < rank; i++) {
        if (full_shape[i] == reduced_shape[i]) {
            continue;
        }
        if (reduced_shape[i] != 1 || full_shape[i] == 1) {
            return -1;
        }
        if (dim >= 0) {
            return -1;
        }
        dim = i;
    }
    return dim;
}

bool LinalgGenericToLinalgSoftmax::isFillOp(FillOp fill_op, uint64_t constant) const {
    if (!fill_op || fill_op.getInputs().size() != 1 || fill_op.getOutputs().size() != 1) {
        return false;
    }
    Value input = fill_op.getInputs()[0];
    arith::ConstantOp constant_op = llvm::dyn_cast_or_null<arith::ConstantOp>(input.getDefiningOp());
    if (!constant_op) {
        return false;
    }
    auto float_attr = llvm::dyn_cast_or_null<FloatAttr>(constant_op.getValue());
    if (!float_attr || std::bit_cast<uint64_t>(float_attr.getValueAsDouble()) != constant) {
        return false;
    }

    return true;
}

/*
 * y[..., d, ...] = exp(x) / sum_d(exp(x_d))
 *
 * tmp00 = [-oo, ..., -oo] (linalg.fill)
 * tmp01 = max_d(tmp00, x) (linalg.generic)
 * tmp02 = tmp01 (tensor.expand_shape)
 * tmp03 = x - tmp02 (linalg.generic)
 * tmp04 = exp(tmp03) (linalg.exp)
 * tmp05 = [0, ..., 0] (linalg.fill)
 * tmp06 = tmp04 + tmp05 (linalg.generic)
 * y = tmp04 / tmp06 (linalg.generic)
 */
LogicalResult LinalgGenericToLinalgSoftmax::matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter) const {
    // y = tmp04 / tmp06 (linalg.generic)
    if (generic_op.getInputs().size() != 2 || generic_op.getOutputs().size() != 1 ||
        !this->isElementwiseRegion<arith::DivFOp>(generic_op.getRegion())) {
        return failure();
    }
    Value y = generic_op.getOutputs()[0];
    Value tmp04 = generic_op.getInputs()[0];
    Value tmp06 = generic_op.getInputs()[1];
    RankedTensorType y_type = llvm::dyn_cast_or_null<RankedTensorType>(y.getType());
    RankedTensorType tmp04_type = llvm::dyn_cast_or_null<RankedTensorType>(tmp04.getType());
    RankedTensorType tmp06_type = llvm::dyn_cast_or_null<RankedTensorType>(tmp06.getType());
    if (!y_type || !tmp04_type || !tmp06_type || !this->equalShapes(y_type.getShape(), tmp04_type.getShape())) {
        return failure();
    }
    ArrayRef<int64_t> full_shape = y_type.getShape();
    ArrayRef<int64_t> reduced_shape = tmp06_type.getShape();
    size_t rank = y_type.getRank();
    long long dim = this->determineDimension(full_shape, reduced_shape);
    if (dim < 0 || !this->checkIndexingMaps(generic_op.getIndexingMaps(), {false, true, false}, dim) ||
        !this->checkIteratorTypes(generic_op.getIteratorTypes(), rank)) {
        return failure();
    }

    // tmp06 = tmp04 + tmp05 (linalg.generic)
    auto tmp06_op = llvm::dyn_cast_or_null<GenericOp>(tmp06.getDefiningOp());
    if (!tmp06_op || tmp06_op.getInputs().size() != 1 || tmp06_op.getOutputs().size() != 1 ||
        !this->isElementwiseRegion<arith::AddFOp>(tmp06_op.getRegion()) ||
        !this->checkIndexingMaps(tmp06_op.getIndexingMaps(), {false, true}, dim) ||
        !this->checkIteratorTypes(tmp06_op.getIteratorTypes(), rank, dim) || tmp06_op.getInputs()[0] != tmp04) {
        return failure();
    }
    Value tmp05 = tmp06_op.getOutputs()[0];
    RankedTensorType tmp05_type = llvm::dyn_cast_or_null<RankedTensorType>(tmp05.getType());
    if (!tmp05_type || !this->equalShapes(tmp05_type.getShape(), reduced_shape)) {
        return failure();
    }

    // tmp05 = [0, ..., 0] (linalg.fill)
    auto tmp05_op = llvm::dyn_cast_or_null<FillOp>(tmp05.getDefiningOp());
    if (!this->isFillOp(tmp05_op, 0)) {
        return failure();
    }

    // tmp04 = exp(tmp03) (linalg.exp)
    auto tmp04_op = llvm::dyn_cast_or_null<ExpOp>(tmp04.getDefiningOp());
    if (!tmp04_op || tmp04_op.getInputs().size() != 1 || tmp04_op.getOutputs().size() != 1) {
        return failure();
    }
    Value tmp03 = tmp04_op.getInputs()[0];
    RankedTensorType tmp03_type = llvm::dyn_cast_or_null<RankedTensorType>(tmp03.getType());
    if (!tmp03_type || !this->equalShapes(tmp03_type.getShape(), full_shape)) {
        return failure();
    }

    // tmp03 = x - tmp02 (linalg.generic)
    auto tmp03_op = llvm::dyn_cast_or_null<GenericOp>(tmp03.getDefiningOp());
    if (!tmp03_op || tmp03_op.getInputs().size() != 2 || tmp03_op.getOutputs().size() != 1 ||
        !this->isElementwiseRegion<arith::SubFOp>(tmp03_op.getRegion()) ||
        !this->checkIndexingMaps(tmp03_op.getIndexingMaps(), {false, true, false}, dim) ||
        !this->checkIteratorTypes(tmp03_op.getIteratorTypes(), rank)) {
        return failure();
    }
    Value x = tmp03_op.getInputs()[0];
    Value tmp02 = tmp03_op.getInputs()[1];
    RankedTensorType x_type = llvm::dyn_cast_or_null<RankedTensorType>(x.getType());
    RankedTensorType tmp02_type = llvm::dyn_cast_or_null<RankedTensorType>(tmp02.getType());
    if (!x_type || !tmp02_type || !this->equalShapes(x_type.getShape(), full_shape) ||
        !this->equalShapes(tmp02_type.getShape(), reduced_shape)) {
        return failure();
    }

    // tmp02 = tmp01 (tensor.expand_shape)
    auto tmp02_op = llvm::dyn_cast_or_null<tensor::ExpandShapeOp>(tmp02.getDefiningOp());
    if (!tmp02_op || !this->equalShapes(tmp02_op.getStaticOutputShape(), reduced_shape)) {
        return failure();
    }
    Value tmp01 = tmp02_op.getSrc();
    RankedTensorType tmp01_type = tmp02_op.getSrcType();
    if (tmp01_type.getRank() + 1 != static_cast<int64_t>(rank)) {
        return failure();
    }
    ArrayRef<int64_t> collapsed_shape = tmp01_type.getShape();
    for (long long i = 0; i < static_cast<long long>(rank); i++) {
        if (i < dim) {
            if (collapsed_shape[i] != full_shape[i]) {
                return failure();
            }
        } else if (i > dim) {
            if (collapsed_shape[i - 1] != full_shape[i]) {
                return failure();
            }
        }
    }

    // tmp01 = max_d(tmp00, x) (linalg.generic)
    auto tmp01_op = llvm::dyn_cast_or_null<GenericOp>(tmp01.getDefiningOp());
    if (!tmp01_op || tmp01_op.getInputs().size() != 1 || tmp01_op.getOutputs().size() != 1 ||
        !this->isElementwiseRegion<arith::MaximumFOp>(tmp01_op.getRegion()) || tmp01_op.getIndexingMaps().size() != 2 ||
        !this->checkIndexingMap(tmp01_op.getIndexingMaps()[0]) ||
        !this->checkIndexingMapCollapsed(tmp01_op.getIndexingMaps()[1], dim) ||
        !this->checkIteratorTypes(tmp01_op.getIteratorTypes(), rank, dim) || tmp01_op.getInputs()[0] != x) {
        return failure();
    }
    Value tmp00 = tmp01_op.getOutputs()[0];
    RankedTensorType tmp00_type = llvm::dyn_cast_or_null<RankedTensorType>(tmp00.getType());
    if (!tmp00_type || !this->equalShapes(tmp00_type.getShape(), collapsed_shape)) {
        return failure();
    }

    // tmp00 = [-oo, ..., -oo] (linalg.fill)
    auto tmp00_op = llvm::dyn_cast_or_null<FillOp>(tmp00.getDefiningOp());
    if (!this->isFillOp(tmp00_op, 0xFFF0000000000000)) {
        return failure();
    }

    rewriter.replaceOpWithNewOp<SoftmaxOp>(generic_op, generic_op.getResultTypes(), x, y, dim);

    return success();
}

} // namespace linalg
} // namespace mlir
