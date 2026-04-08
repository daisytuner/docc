#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace linalg {

/**
 * y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
 */
struct LinalgGenericToLinalgCustomBatchNorm2DNchw : public OpRewritePattern<GenericOp> {
    using OpRewritePattern<GenericOp>::OpRewritePattern;

    bool is4dIdentityMap(Attribute attr) const;
    bool is3dIdentityMap(Attribute attr) const;
    bool checkIndexingMaps(ArrayAttr array_attr) const;

    bool checkIteratorTypes(ArrayAttr array_attr) const;

    template<typename ElemOp>
    bool isElementwiseRegion(Region& region) const {
        if (region.getBlocks().size() != 1) {
            return false;
        }
        auto& block = region.getBlocks().front();
        return (
            block.getOperations().size() == 2 && llvm::dyn_cast_or_null<ElemOp>(block.getOperations().front()) &&
            llvm::dyn_cast_or_null<YieldOp>(block.getOperations().back())
        );
    }

    template<typename ElemOp>
    bool checkGenericOp(GenericOp generic_op) const {
        return (
            generic_op.getInputs().size() == 2 && generic_op.getOutputs().size() == 1 &&
            this->checkIndexingMaps(generic_op.getIndexingMaps()) &&
            this->checkIteratorTypes(generic_op.getIteratorTypes()) &&
            this->isElementwiseRegion<ElemOp>(generic_op.getRegion())
        );
    }

    bool isTypeNCHW(Type type, Type element_type, int64_t n, int64_t c, int64_t h, int64_t w) const;
    bool isTypeC11(Type type, Type element_type, int64_t c) const;
    bool isTypeC(Type type, Type element_type, int64_t c) const;

    bool checkExpandShape(tensor::ExpandShapeOp expand_shape_op, int64_t c) const;

    LogicalResult matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter) const override;
};

} // namespace linalg
} // namespace mlir
