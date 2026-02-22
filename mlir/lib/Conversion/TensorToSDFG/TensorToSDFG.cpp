#include "mlir/Conversion/TensorToSDFG/TensorToSDFG.h"

#include <llvm/Support/LogicalResult.h>

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTTENSORTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace tensor2sdfg {

struct EmptyOpConversion : public OpRewritePattern<tensor::EmptyOp> {
    using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::EmptyOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        Type resultType = op.getResult().getType();

        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({resultType}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        sdfg::MallocOp malloc_op = rewriter.create<sdfg::MallocOp>(block_op.getLoc(), resultType);

        block_op.getBody().front().back().setOperands(SmallVector<Value>({malloc_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

} // namespace tensor2sdfg

namespace {

struct ConvertTensorToSDFG : public impl::ConvertTensorToSDFGBase<ConvertTensorToSDFG> {
    using ConvertTensorToSDFGBase::ConvertTensorToSDFGBase;

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&this->getContext());
        tensor::populateTensorToSDFGPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(this->getOperation(), std::move(patterns)))) {
            this->signalPassFailure();
        }
    }
};

} // namespace

namespace tensor {

void populateTensorToSDFGPatterns(RewritePatternSet& patterns) {
    patterns.add<tensor2sdfg::EmptyOpConversion>(patterns.getContext());
}

} // namespace tensor
} // namespace mlir
