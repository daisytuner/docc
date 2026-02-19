#include "mlir/Conversion/LinalgToSDFG/LinalgToSDFG.h"

#include <llvm/Support/LogicalResult.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTLINALGTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace linalg2sdfg {

struct FillOpConversion : public OpRewritePattern<linalg::FillOp> {
    using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::FillOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // TODO Check types

        Value value = op.value();
        Value result = op.result();

        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result.getType()}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        Location input_loc = block_op.getLoc();
        Value input_value;
        if (auto constant_op = dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
            sdfg::ConstantOp input_op =
                rewriter.create<sdfg::ConstantOp>(block_op.getLoc(), constant_op.getType(), constant_op.getValue());
            input_value = input_op.getResult();
            input_loc = input_op.getLoc();
        } else {
            sdfg::MemletOp input_op = rewriter.create<sdfg::MemletOp>(block_op.getLoc(), value.getType(), value);
            input_value = input_op.getResult();
            input_loc = input_op.getLoc();
        }
        sdfg::FillOp fill_op = rewriter.create<sdfg::FillOp>(input_loc, result.getType(), input_value);
        sdfg::MemletOp output_memlet_op = rewriter.create<sdfg::MemletOp>(fill_op.getLoc(), result.getType(), fill_op);

        block_op.getBody().front().back().setOperands(SmallVector<Value>({output_memlet_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

struct MatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // Expect 2 inputs and 1 ouput
        if (op.getInputs().size() != 2 || op.getOutputs().size() != 1 || op.getResultTensors().size() != 1) {
            return failure();
        }

        // For now, ignore broadcast and transpose
        if (op->hasAttr("indexing_maps")) {
            return failure();
        }

        // TODO Check types

        Value lhs = op.getInputs()[0];
        Value rhs = op.getInputs()[1];
        Value output = op.getOutputs()[0];
        Value result = op.getResultTensors()[0];

        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result.getType()}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        sdfg::MemletOp res_input_memlet_op =
            rewriter.create<sdfg::MemletOp>(block_op.getLoc(), output.getType(), output);
        sdfg::MemletOp lhs_memlet_op =
            rewriter.create<sdfg::MemletOp>(res_input_memlet_op.getLoc(), lhs.getType(), lhs);
        sdfg::MemletOp rhs_memlet_op = rewriter.create<sdfg::MemletOp>(lhs_memlet_op.getLoc(), rhs.getType(), rhs);
        sdfg::MatmulOp matmul_op = rewriter.create<
            sdfg::MatmulOp>(rhs_memlet_op.getLoc(), result.getType(), res_input_memlet_op, lhs_memlet_op, rhs_memlet_op);
        sdfg::MemletOp output_memlet_op =
            rewriter.create<sdfg::MemletOp>(matmul_op.getLoc(), result.getType(), matmul_op);

        block_op.getBody().front().back().setOperands({output_memlet_op});
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

struct TransposeOpConversion : public OpRewritePattern<linalg::TransposeOp> {
    using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::TransposeOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // Expect 1 input and 1 output
        if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) {
            return failure();
        }

        Value input = op.getInput();
        Value result = op.getResult()[0];

        // Get permutation from linalg::TransposeOp
        ArrayRef<int64_t> permutation = op.getPermutation();

        // Create block containing the transpose memlet operation
        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result.getType()}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        // Create input memlet
        sdfg::MemletOp input_memlet_op = rewriter.create<sdfg::MemletOp>(block_op.getLoc(), input.getType(), input);

        // Create transpose memlet with permutation
        sdfg::TransposeMemletOp transpose_op = rewriter.create<sdfg::TransposeMemletOp>(
            input_memlet_op.getLoc(), result.getType(), input_memlet_op, rewriter.getDenseI64ArrayAttr(permutation)
        );

        // Create output memlet
        sdfg::MemletOp output_memlet_op =
            rewriter.create<sdfg::MemletOp>(transpose_op.getLoc(), result.getType(), transpose_op);

        block_op.getBody().front().back().setOperands({output_memlet_op});
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

} // namespace linalg2sdfg

namespace {

struct ConvertLinalgToSDFG : public impl::ConvertLinalgToSDFGBase<ConvertLinalgToSDFG> {
    using ConvertLinalgToSDFGBase::ConvertLinalgToSDFGBase;

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&this->getContext());
        linalg::populateLinalgToSDFGPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(this->getOperation(), std::move(patterns)))) {
            this->signalPassFailure();
        }
    }
};

} // namespace

namespace linalg {

void populateLinalgToSDFGPatterns(RewritePatternSet& patterns) {
    patterns.add<
        linalg2sdfg::FillOpConversion,
        linalg2sdfg::MatmulOpConversion,
        linalg2sdfg::TransposeOpConversion>(patterns.getContext());
}

} // namespace linalg
} // namespace mlir
