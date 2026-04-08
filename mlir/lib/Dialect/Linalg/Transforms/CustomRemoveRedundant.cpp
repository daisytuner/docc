#include <cstdint>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CustomTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGCUSTOMREMOVEREDUNDANTOPSPASS
#include "mlir/Dialect/Linalg/CustomPasses.h.inc"

namespace linalg {

template<typename Op2D, uint64_t InitValue>
struct Linalg2DNchwRemoveLinalgFill : public OpRewritePattern<Op2D> {
    using OpRewritePattern<Op2D>::OpRewritePattern;

    LogicalResult matchAndRewrite(Op2D conv_op, PatternRewriter& rewriter) const override {
        // Only accept one output
        if (conv_op.getOutputs().size() != 1) {
            return failure();
        }

        // Get fill op
        auto fill_op = llvm::dyn_cast_or_null<FillOp>(conv_op.getOutputs()[0].getDefiningOp());
        if (!fill_op) {
            return failure();
        }
        if (fill_op.getOutputs().size() != 1) {
            return failure();
        }

        // Check that dilations are 1
        for (auto dilation : conv_op.getDilations()) {
            if (dilation.getSExtValue() != 1) {
                return failure();
            }
        }

        // Check that the fill op initializes with zero
        if (fill_op.getInputs().size() != 1) {
            return failure();
        }
        auto constant_op = llvm::dyn_cast_or_null<arith::ConstantOp>(fill_op.getInputs()[0].getDefiningOp());
        if (!constant_op) {
            return failure();
        }
        if (auto float_attr = llvm::dyn_cast<FloatAttr>(constant_op.getValue())) {
            if (std::bit_cast<uint64_t>(float_attr.getValueAsDouble()) != InitValue) {
                return failure();
            }
        } else if (auto integer_attr = llvm::dyn_cast<IntegerAttr>(constant_op.getValue())) {
            if (integer_attr.getUInt() != InitValue) {
                return failure();
            }
        } else {
            return failure();
        }

        // Set output of convolution direclty to output of fill op
        conv_op.getOutputsMutable()[0].assign(fill_op.getOutputs()[0]);

        return success();
    }
};

struct LinalgCustomRemoveRedundantOpsPass
    : public impl::LinalgCustomRemoveRedundantOpsPassBase<LinalgCustomRemoveRedundantOpsPass> {
    using LinalgCustomRemoveRedundantOpsPassBase::LinalgCustomRemoveRedundantOpsPassBase;

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&this->getContext());
        populateLinalgCustomRemoveRedundantOpsPass(patterns);
        if (failed(applyPatternsAndFoldGreedily(this->getOperation(), std::move(patterns)))) {
            this->signalPassFailure();
        }
    }
};

void populateLinalgCustomRemoveRedundantOpsPass(RewritePatternSet& patterns) {
    patterns.add<
        Linalg2DNchwRemoveLinalgFill<Conv2DNchwFchwOp, 0x0000000000000000>,
        Linalg2DNchwRemoveLinalgFill<PoolingNchwMaxOp, 0xFFF0000000000000>,
        Linalg2DNchwRemoveLinalgFill<PoolingNchwSumOp, 0x0000000000000000>>(patterns.getContext());
}

} // namespace linalg
} // namespace mlir
