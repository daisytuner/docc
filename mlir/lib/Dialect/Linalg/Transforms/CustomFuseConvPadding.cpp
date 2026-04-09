#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgCustomOps.h"
#include "mlir/Dialect/Linalg/Transforms/CustomTransforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGCUSTOMFUSECONVPADDINGPASS
#include "mlir/Dialect/Linalg/CustomPasses.h.inc"

namespace linalg {

struct FusePaddingIntoConv2DNchwFchw : OpRewritePattern<Conv2DNchwFchwOp> {
    using OpRewritePattern<Conv2DNchwFchwOp>::OpRewritePattern;

    bool broadcastOpIsConvBias(BroadcastOp broadcast_op) const {
        auto dims = broadcast_op.getDimensions();
        if (dims.size() != 3 || dims[0] != 0 || dims[1] != 2 || dims[2] != 3) {
            return false;
        }

        auto biasShape = broadcast_op.getInput().getType().getShape();
        auto outputShape = broadcast_op.getInit().getType().getShape();
        if (biasShape.size() != 1 || outputShape.size() != 4 || biasShape[0] != outputShape[1]) {
            return false;
        }

        return true;
    }

    LogicalResult matchAndRewrite(Conv2DNchwFchwOp conv_op, PatternRewriter& rewriter) const override {
        // Check and get values
        if (conv_op.getInputs().size() != 2) {
            return failure();
        }
        if (conv_op.getOutputs().size() != 1) {
            return failure();
        }
        if (conv_op.getResults().size() != 1) {
            return failure();
        }
        Type resultType = conv_op.getResults()[0].getType();
        Value input = conv_op.getInputs()[0];
        Value weights = conv_op.getInputs()[1];
        Value bias;
        Value output = conv_op.getOutputs()[0];

        // Check padding operation
        auto pad_op = llvm::dyn_cast_or_null<tensor::PadOp>(input.getDefiningOp());
        if (!pad_op || pad_op.getStaticLow().size() != 4 || pad_op.getStaticLow()[0] != 0 ||
            pad_op.getStaticLow()[1] != 0 || pad_op.getStaticLow()[2] == INT64_MIN ||
            pad_op.getStaticLow()[3] == INT64_MIN || pad_op.getStaticHigh().size() != 4 ||
            pad_op.getStaticHigh()[0] != 0 || pad_op.getStaticHigh()[1] != 0 ||
            pad_op.getStaticHigh()[2] == INT64_MIN || pad_op.getStaticHigh()[3] == INT64_MIN ||
            pad_op.getRegion().getBlocks().size() != 1) {
            return failure();
        }
        auto& block = pad_op.getRegion().getBlocks().front();
        if (block.getOperations().size() != 1) {
            return failure();
        }
        auto yield_op = llvm::dyn_cast_or_null<tensor::YieldOp>(block.getOperations().front());
        if (!yield_op) {
            return failure();
        }

        // Check that padding is performed with a zero
        auto constant_op = llvm::dyn_cast_or_null<arith::ConstantOp>(yield_op.getValue().getDefiningOp());
        if (!constant_op) {
            return failure();
        }
        if (auto float_attr = llvm::dyn_cast<FloatAttr>(constant_op.getValue())) {
            if (float_attr.getValueAsDouble() != 0.0) {
                return failure();
            }
        } else if (auto integer_attr = llvm::dyn_cast<IntegerAttr>(constant_op.getValue())) {
            if (integer_attr.getUInt() != 0) {
                return failure();
            }
        } else {
            return failure();
        }

        // Remap the input to the un-padded value
        input = pad_op.getSource();

        // Check if the output is a broadcasted value
        if (auto broadcast_op = llvm::dyn_cast_or_null<BroadcastOp>(output.getDefiningOp())) {
            if (this->broadcastOpIsConvBias(broadcast_op)) {
                // Add bias & remap output
                bias = broadcast_op.getInput();
                output = broadcast_op.getInit();
            }
        }

        // Copy the padding values
        SmallVector<int64_t> paddings;
        paddings.push_back(pad_op.getStaticLow()[2]);
        paddings.push_back(pad_op.getStaticLow()[3]);
        paddings.push_back(pad_op.getStaticHigh()[2]);
        paddings.push_back(pad_op.getStaticHigh()[3]);

        // Copy strides and dilations
        SmallVector<int64_t> strides, dilations;
        for (int64_t stride : conv_op.getStrides().getValues<int64_t>()) {
            strides.push_back(stride);
        }
        for (int64_t dilation : conv_op.getDilations().getValues<int64_t>()) {
            dilations.push_back(dilation);
        }
        auto strides_attr = DenseI64ArrayAttr::get(this->getContext(), strides);
        auto dilations_attr = DenseI64ArrayAttr::get(this->getContext(), dilations);
        auto paddings_attr = DenseI64ArrayAttr::get(this->getContext(), paddings);

        // Create the custom convolution with paddings
        rewriter.replaceOpWithNewOp<custom::Conv2DNchwFchwOp>(
            conv_op, resultType, input, weights, bias, output, strides_attr, dilations_attr, paddings_attr
        );

        return success();
    }
};

struct LinalgCustomFuseConvPaddingPass
    : public impl::LinalgCustomFuseConvPaddingPassBase<LinalgCustomFuseConvPaddingPass> {
    using LinalgCustomFuseConvPaddingPassBase::LinalgCustomFuseConvPaddingPassBase;

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&this->getContext());
        populateLinalgCustomFuseConvPaddingPass(patterns);
        if (failed(applyPatternsAndFoldGreedily(this->getOperation(), std::move(patterns)))) {
            this->signalPassFailure();
        }
    }
};

void populateLinalgCustomFuseConvPaddingPass(RewritePatternSet& patterns) {
    patterns.add<FusePaddingIntoConv2DNchwFchw>(patterns.getContext());
}

} // namespace linalg
} // namespace mlir
