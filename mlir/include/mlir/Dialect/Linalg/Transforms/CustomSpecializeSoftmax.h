#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace linalg {

/**
 * y[..., d, ...] = exp(x) / sum_d(exp(x_d))
 */
struct LinalgGenericToLinalgSoftmax : public OpRewritePattern<GenericOp> {
    using OpRewritePattern<GenericOp>::OpRewritePattern;

    bool checkIndexingMap(Attribute attr, long long dim = -1) const;
    bool checkIndexingMapCollapsed(Attribute attr, long long dim) const;
    bool checkIndexingMaps(ArrayAttr maps, const std::vector<bool>& reduced_mask, long long dim) const;

    bool checkIteratorTypes(ArrayAttr types, size_t rank, long long dim = -1) const;

    bool equalShapes(const ArrayRef<int64_t>& shape1, const ArrayRef<int64_t>& shape2) const;
    long long determineDimension(const ArrayRef<int64_t>& full_shape, const ArrayRef<int64_t>& reduced_shape) const;

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

    bool isFillOp(FillOp fill_op, uint64_t constant) const;

    LogicalResult matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter) const override;
};

} // namespace linalg
} // namespace mlir
