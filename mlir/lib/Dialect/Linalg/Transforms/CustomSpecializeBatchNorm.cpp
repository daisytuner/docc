#include "mlir/Dialect/Linalg/Transforms/CustomSpecializeBatchNorm.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgCustomOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace linalg {

bool LinalgGenericToLinalgCustomBatchNorm2DNchw::is4dIdentityMap(Attribute attr) const {
    auto affine_map_attr = llvm::dyn_cast_or_null<AffineMapAttr>(attr);
    if (!affine_map_attr) {
        return false;
    }
    AffineMap affine_map = affine_map_attr.getAffineMap();
    return (affine_map.getNumInputs() == 4 && affine_map.isIdentity());
}

bool LinalgGenericToLinalgCustomBatchNorm2DNchw::is3dIdentityMap(Attribute attr) const {
    auto affine_map_attr = llvm::dyn_cast_or_null<AffineMapAttr>(attr);
    if (!affine_map_attr) {
        return false;
    }
    AffineMap affine_map = affine_map_attr.getAffineMap();
    if (affine_map.getNumInputs() != 4 || affine_map.getNumResults() != 3) {
        return false;
    }
    AffineDimExpr affine_dim_expr = llvm::dyn_cast<AffineDimExpr>(affine_map.getResult(0));
    if (!affine_dim_expr || affine_dim_expr.getPosition() != 1) {
        return false;
    }
    AffineConstantExpr affine_const_expr_1 = llvm::dyn_cast<AffineConstantExpr>(affine_map.getResult(1));
    if (!affine_const_expr_1 || affine_const_expr_1.getValue() != 0) {
        return false;
    }
    AffineConstantExpr affine_const_expr_2 = llvm::dyn_cast<AffineConstantExpr>(affine_map.getResult(2));
    if (!affine_const_expr_2 || affine_const_expr_2.getValue() != 0) {
        return false;
    }
    return true;
}

bool LinalgGenericToLinalgCustomBatchNorm2DNchw::checkIndexingMaps(ArrayAttr array_attr) const {
    return array_attr.size() == 3 && this->is4dIdentityMap(array_attr[0]) && this->is3dIdentityMap(array_attr[1]) &&
           this->is4dIdentityMap(array_attr[2]);
}

bool LinalgGenericToLinalgCustomBatchNorm2DNchw::checkIteratorTypes(ArrayAttr array_attr) const {
    for (auto attr : array_attr) {
        auto iterator_type_attr = llvm::dyn_cast_or_null<IteratorTypeAttr>(attr);
        if (!iterator_type_attr) {
            return false;
        }
        if (iterator_type_attr.getValue() != utils::IteratorType::parallel) {
            return false;
        }
    }
    return true;
}

bool LinalgGenericToLinalgCustomBatchNorm2DNchw::
    isTypeNCHW(Type type, Type element_type, int64_t n, int64_t c, int64_t h, int64_t w) const {
    auto ranked_tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(type);
    return (
        ranked_tensor_type && ranked_tensor_type.getElementType() == element_type &&
        ranked_tensor_type.getRank() == 4 && ranked_tensor_type.getDimSize(0) == n &&
        ranked_tensor_type.getDimSize(1) == c && ranked_tensor_type.getDimSize(2) == h &&
        ranked_tensor_type.getDimSize(3) == w
    );
}

bool LinalgGenericToLinalgCustomBatchNorm2DNchw::isTypeC11(Type type, Type element_type, int64_t c) const {
    auto ranked_tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(type);
    return (
        ranked_tensor_type && ranked_tensor_type.getElementType() == element_type &&
        ranked_tensor_type.getRank() == 3 && ranked_tensor_type.getDimSize(0) == c &&
        ranked_tensor_type.getDimSize(1) == 1 && ranked_tensor_type.getDimSize(2) == 1
    );
}

bool LinalgGenericToLinalgCustomBatchNorm2DNchw::isTypeC(Type type, Type element_type, int64_t c) const {
    auto ranked_tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(type);
    return (
        ranked_tensor_type && ranked_tensor_type.getElementType() == element_type &&
        ranked_tensor_type.getRank() == 1 && ranked_tensor_type.getDimSize(0) == c
    );
}

bool LinalgGenericToLinalgCustomBatchNorm2DNchw::checkExpandShape(tensor::ExpandShapeOp expand_shape_op, int64_t c)
    const {
    if (expand_shape_op.getReassociation().size() != 1 || expand_shape_op.getStaticOutputShape().size() != 3 ||
        expand_shape_op.getStaticOutputShape()[0] != c || expand_shape_op.getStaticOutputShape()[1] != 1 ||
        expand_shape_op.getStaticOutputShape()[2] != 1 || expand_shape_op.getOutputShape().size() != 0) {
        return false;
    }
    ArrayAttr reassociation = llvm::dyn_cast_or_null<ArrayAttr>(expand_shape_op.getReassociation()[0]);
    if (!reassociation || reassociation.size() != 3) {
        return false;
    }
    IntegerAttr reassociation_0 = llvm::dyn_cast_or_null<IntegerAttr>(reassociation[0]);
    IntegerAttr reassociation_1 = llvm::dyn_cast_or_null<IntegerAttr>(reassociation[1]);
    IntegerAttr reassociation_2 = llvm::dyn_cast_or_null<IntegerAttr>(reassociation[2]);
    return (
        reassociation_0 && reassociation_0.getInt() == 0 && reassociation_1 && reassociation_1.getInt() == 1 &&
        reassociation_2 && reassociation_2.getInt() == 2
    );
}

/*
 * y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 *
 * tmp01 = Var[x] + eps (linalg.add)
 * tmp02 = sqrt(tmp01) (linalg.sqrt)
 * tmp03 = 1 / tmp02 (linalg.div)
 * tmp04 = E[x] (tensor.expand_shape)
 * tmp05 = tmp03 (tensor.expand_shape)
 * tmp06 = x - tmp04 (linalg.generic)
 * tmp07 = tmp06 * tmp05 (linalg.generic)
 * tmp08 = gamma (tensor.expand_shape)
 * tmp09 = tmp07 * tmp08 (linalg.generic)
 * tmp10 = beta (tensor.expand_shape)
 * y = tmp09 + tmp10 (linalg.generic)
 */
LogicalResult LinalgGenericToLinalgCustomBatchNorm2DNchw::matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter)
    const {
    // y = tmp09 + tmp10 (linalg.generic)
    if (!this->checkGenericOp<arith::AddFOp>(generic_op)) {
        return failure();
    }
    Value tmp09 = generic_op.getInputs()[0];
    Value tmp10 = generic_op.getInputs()[1];
    RankedTensorType tmp09_type = llvm::dyn_cast_or_null<RankedTensorType>(tmp09.getType());
    if (!tmp09_type || tmp09_type.getRank() != 4) {
        return failure();
    }
    int64_t n = tmp09_type.getDimSize(0);
    int64_t c = tmp09_type.getDimSize(1);
    int64_t h = tmp09_type.getDimSize(2);
    int64_t w = tmp09_type.getDimSize(3);
    Type element_type = tmp09_type.getElementType();
    if (!this->isTypeNCHW(generic_op.getOutputs()[0].getType(), element_type, n, c, h, w) ||
        !this->isTypeC11(tmp10.getType(), element_type, c)) {
        return failure();
    }

    // tmp10 = beta (tensor.expand_shape)
    auto tmp10_op = llvm::dyn_cast_or_null<tensor::ExpandShapeOp>(tmp10.getDefiningOp());
    if (!tmp10_op || !this->checkExpandShape(tmp10_op, c)) {
        return failure();
    }
    Value beta = tmp10_op.getSrc();
    if (!this->isTypeC(beta.getType(), element_type, c)) {
        return failure();
    }

    // tmp09 = tmp07 * tmp08 (linalg.generic)
    auto tmp09_op = llvm::dyn_cast_or_null<GenericOp>(tmp09.getDefiningOp());
    if (!tmp09_op || !this->checkGenericOp<arith::MulFOp>(tmp09_op)) {
        return failure();
    }
    Value tmp07 = tmp09_op.getInputs()[0];
    Value tmp08 = tmp09_op.getInputs()[1];
    if (!this->isTypeNCHW(tmp07.getType(), element_type, n, c, h, w) ||
        !this->isTypeC11(tmp08.getType(), element_type, c)) {
        return failure();
    }

    // tmp08 = gamma (tensor.expand_shape)
    auto tmp08_op = llvm::dyn_cast_or_null<tensor::ExpandShapeOp>(tmp08.getDefiningOp());
    if (!tmp08_op || !this->checkExpandShape(tmp08_op, c)) {
        return failure();
    }
    Value gamma = tmp08_op.getSrc();
    if (!this->isTypeC(gamma.getType(), element_type, c)) {
        return failure();
    }

    // tmp07 = tmp06 * tmp05 (linalg.generic)
    auto tmp07_op = llvm::dyn_cast_or_null<GenericOp>(tmp07.getDefiningOp());
    if (!tmp07_op || !this->checkGenericOp<arith::MulFOp>(tmp07_op)) {
        return failure();
    }
    Value tmp06 = tmp07_op.getInputs()[0];
    Value tmp05 = tmp07_op.getInputs()[1];
    if (!this->isTypeNCHW(tmp06.getType(), element_type, n, c, h, w) ||
        !this->isTypeC11(tmp05.getType(), element_type, c)) {
        return failure();
    }

    // tmp06 = x - tmp04 (linalg.generic)
    auto tmp06_op = llvm::dyn_cast_or_null<GenericOp>(tmp06.getDefiningOp());
    if (!tmp06_op || !this->checkGenericOp<arith::SubFOp>(tmp06_op)) {
        return failure();
    }
    Value x = tmp06_op.getInputs()[0];
    Value tmp04 = tmp06_op.getInputs()[1];
    if (!this->isTypeNCHW(x.getType(), element_type, n, c, h, w) ||
        !this->isTypeC11(tmp04.getType(), element_type, c)) {
        return failure();
    }

    // tmp05 = tmp03 (tensor.expand_shape)
    auto tmp05_op = llvm::dyn_cast_or_null<tensor::ExpandShapeOp>(tmp05.getDefiningOp());
    if (!tmp05_op || !this->checkExpandShape(tmp05_op, c)) {
        return failure();
    }
    Value tmp03 = tmp05_op.getSrc();
    if (!this->isTypeC(tmp03.getType(), element_type, c)) {
        return failure();
    }

    // tmp04 = E[x] (tensor.expand_shape)
    auto tmp04_op = llvm::dyn_cast_or_null<tensor::ExpandShapeOp>(tmp04.getDefiningOp());
    if (!tmp04_op || !this->checkExpandShape(tmp04_op, c)) {
        return failure();
    }
    Value e = tmp04_op.getSrc();
    if (!this->isTypeC(e.getType(), element_type, c)) {
        return failure();
    }

    // tmp03 = 1 / tmp02 (linalg.div)
    auto tmp03_op = llvm::dyn_cast_or_null<DivOp>(tmp03.getDefiningOp());
    if (!tmp03_op || tmp03_op.getInputs().size() != 2 || tmp03_op.getOutputs().size() != 1) {
        return failure();
    }
    auto constant_one = llvm::dyn_cast_or_null<arith::ConstantOp>(tmp03_op.getInputs()[0].getDefiningOp());
    if (!constant_one) {
        return failure();
    }
    if (auto constant_one_int_attr = llvm::dyn_cast_or_null<DenseIntElementsAttr>(constant_one.getValue())) {
        if (constant_one_int_attr.getNumElements() != 1 || (*constant_one_int_attr.begin()).getSExtValue() != 1) {
            return failure();
        }
    } else if (auto constant_one_float_attr = llvm::dyn_cast_or_null<DenseFPElementsAttr>(constant_one.getValue())) {
        if (constant_one_float_attr.getNumElements() != 1 ||
            (*constant_one_float_attr.begin()).convertToDouble() != 1.0) {
            return failure();
        }
    } else {
        return failure();
    }
    Value tmp02 = tmp03_op.getInputs()[1];
    if (!this->isTypeC(tmp02.getType(), element_type, c)) {
        return failure();
    }

    // tmp02 = sqrt(tmp01) (linalg.sqrt)
    auto tmp02_op = llvm::dyn_cast_or_null<SqrtOp>(tmp02.getDefiningOp());
    if (!tmp02_op || tmp02_op.getInputs().size() != 1 || tmp02_op.getOutputs().size() != 1) {
        return failure();
    }
    Value tmp01 = tmp02_op.getInputs()[0];
    if (!this->isTypeC(tmp01.getType(), element_type, c)) {
        return failure();
    }

    // tmp01 = Var[x] + eps (linalg.add)
    auto tmp01_op = llvm::dyn_cast_or_null<AddOp>(tmp01.getDefiningOp());
    if (!tmp01_op || tmp01_op.getInputs().size() != 2 || tmp01_op.getOutputs().size() != 1) {
        return failure();
    }
    Value var = tmp01_op.getInputs()[0];
    Value eps = tmp01_op.getInputs()[1];
    auto eps_type = llvm::dyn_cast_or_null<RankedTensorType>(eps.getType());
    if (!this->isTypeC(var.getType(), element_type, c) || !eps_type || eps_type.getElementType() != element_type ||
        eps_type.getRank() != 0) {
        return failure();
    }

    rewriter.replaceOpWithNewOp<linalg::custom::BatchNorm2DNchwOp>(
        generic_op, generic_op.getResults()[0].getType(), x, e, var, gamma, beta, generic_op.getOutputs()[0], eps
    );

    return failure();
}

} // namespace linalg
} // namespace mlir
