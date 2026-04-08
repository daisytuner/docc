#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CustomTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGCUSTOMREMOVEREDUNDANTOPSPASS
#include "mlir/Dialect/Linalg/CustomPasses.h.inc"

namespace linalg {

struct LinalgConv2DNchwFchwRemoveLinalgFill : public OpRewritePattern<Conv2DNchwFchwOp> {
    using OpRewritePattern<Conv2DNchwFchwOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(Conv2DNchwFchwOp conv_op, PatternRewriter& rewriter) const override {
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
        for (auto dilation : conv_op.getDilations().getValues<int64_t>()) {
            if (dilation != 1) {
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
            if (float_attr.getValueAsDouble() != 0.0) {
                return failure();
            }
        } else if (auto integer_attr = llvm::dyn_cast<IntegerAttr>(constant_op.getValue())) {
            if (integer_attr.getInt() != 0) {
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
    patterns.add<LinalgConv2DNchwFchwRemoveLinalgFill>(patterns.getContext());
}

} // namespace linalg
} // namespace mlir
