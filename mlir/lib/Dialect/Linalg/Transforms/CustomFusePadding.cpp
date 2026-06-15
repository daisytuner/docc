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
#define GEN_PASS_DEF_LINALGCUSTOMFUSEPADDINGPASS
#include "mlir/Dialect/Linalg/CustomPasses.h.inc"

namespace linalg {

template<typename Conv2DNchwOp>
struct FusePaddingIntoConv2DNchwBase : OpRewritePattern<Conv2DNchwOp> {
    using OpRewritePattern<Conv2DNchwOp>::OpRewritePattern;

    bool paddingOpHasValue(tensor::PadOp pad_op, double value_as_float, uint64_t value_as_int) const {
        if (!pad_op || pad_op.getStaticLow().size() != 4 || pad_op.getStaticLow()[0] != 0 ||
            pad_op.getStaticLow()[1] != 0 || pad_op.getStaticLow()[2] == INT64_MIN ||
            pad_op.getStaticLow()[3] == INT64_MIN || pad_op.getStaticHigh().size() != 4 ||
            pad_op.getStaticHigh()[0] != 0 || pad_op.getStaticHigh()[1] != 0 ||
            pad_op.getStaticHigh()[2] == INT64_MIN || pad_op.getStaticHigh()[3] == INT64_MIN ||
            pad_op.getRegion().getBlocks().size() != 1) {
            return false;
        }
        auto& block = pad_op.getRegion().getBlocks().front();
        if (block.getOperations().size() != 1) {
            return false;
        }
        auto yield_op = llvm::dyn_cast_or_null<tensor::YieldOp>(block.getOperations().front());
        if (!yield_op) {
            return false;
        }

        // Check that padding is performed with specified value
        auto constant_op = llvm::dyn_cast_or_null<arith::ConstantOp>(yield_op.getValue().getDefiningOp());
        if (!constant_op) {
            return false;
        }
        if (auto float_attr = llvm::dyn_cast<FloatAttr>(constant_op.getValue())) {
            return (float_attr.getValueAsDouble() == value_as_float);
        } else if (auto integer_attr = llvm::dyn_cast<IntegerAttr>(constant_op.getValue())) {
            return (integer_attr.getUInt() == value_as_int);
        } else {
            return false;
        }
    }

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
};

struct FusePaddingIntoConv2DNchwFchw : FusePaddingIntoConv2DNchwBase<Conv2DNchwFchwOp> {
    using FusePaddingIntoConv2DNchwBase<Conv2DNchwFchwOp>::FusePaddingIntoConv2DNchwBase;

    LogicalResult matchAndRewrite(Conv2DNchwFchwOp conv_op, PatternRewriter& rewriter) const override {
        // Check and get values
        if (conv_op.getInputs().size() != 2 || conv_op.getOutputs().size() != 1 || conv_op.getResults().size() != 1) {
            return failure();
        }
        Type resultType = conv_op.getResults()[0].getType();
        Value input = conv_op.getInputs()[0];
        Value weights = conv_op.getInputs()[1];
        Value bias;
        Value output = conv_op.getOutputs()[0];

        // Check padding operation is performed with zero
        auto pad_op = llvm::dyn_cast_or_null<tensor::PadOp>(input.getDefiningOp());
        if (!paddingOpHasValue(pad_op, 0.0, 0ul)) {
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
        auto groups_attr = IntegerAttr::get(IntegerType::get(this->getContext(), 64), 1);

        // Create the custom convolution with paddings
        rewriter.replaceOpWithNewOp<custom::Conv2DNchwFchwOp>(
            conv_op, resultType, input, weights, bias, output, strides_attr, dilations_attr, paddings_attr, groups_attr
        );

        return success();
    }
};

struct FusePaddingIntoDepthwiseConv2DNchwChwOp : FusePaddingIntoConv2DNchwBase<DepthwiseConv2DNchwChwOp> {
    using FusePaddingIntoConv2DNchwBase<DepthwiseConv2DNchwChwOp>::FusePaddingIntoConv2DNchwBase;

    LogicalResult matchAndRewrite(DepthwiseConv2DNchwChwOp conv_op, PatternRewriter& rewriter) const override {
        // Check and get values
        if (conv_op.getInputs().size() != 2 || conv_op.getOutputs().size() != 1 || conv_op.getResults().size() != 1) {
            return failure();
        }
        Type resultType = conv_op.getResults()[0].getType();
        Value input = conv_op.getInputs()[0];
        Value weights = conv_op.getInputs()[1];
        Value bias;
        Value output = conv_op.getOutputs()[0];

        // Check that weights are collapsed
        auto collapse_shape_op = llvm::dyn_cast_or_null<tensor::CollapseShapeOp>(weights.getDefiningOp());
        if (!collapse_shape_op || collapse_shape_op.getSrcType().getRank() != 4 ||
            collapse_shape_op.getResultType().getRank() != 3 ||
            collapse_shape_op.getSrcType().getDimSize(0) != collapse_shape_op.getResultType().getDimSize(0) ||
            collapse_shape_op.getSrcType().getDimSize(2) != collapse_shape_op.getResultType().getDimSize(1) ||
            collapse_shape_op.getSrcType().getDimSize(3) != collapse_shape_op.getResultType().getDimSize(2) ||
            collapse_shape_op.getSrcType().getDimSize(1) != 1) {
            return failure();
        }

        // Remap the weights to the un-collapsed value
        weights = collapse_shape_op.getSrc();

        // Check padding operation is performed with zero
        SmallVector<int64_t> paddings;
        auto pad_op = llvm::dyn_cast_or_null<tensor::PadOp>(input.getDefiningOp());
        if (paddingOpHasValue(pad_op, 0.0, 0ul)) {
            // Remap the input to the un-padded value
            input = pad_op.getSource();

            // Copy the padding values
            paddings.push_back(pad_op.getStaticLow()[2]);
            paddings.push_back(pad_op.getStaticLow()[3]);
            paddings.push_back(pad_op.getStaticHigh()[2]);
            paddings.push_back(pad_op.getStaticHigh()[3]);
        } else {
            // Default padding values
            paddings.push_back(0);
            paddings.push_back(0);
            paddings.push_back(0);
            paddings.push_back(0);
        }

        // Check if the output is a broadcasted value
        if (auto broadcast_op = llvm::dyn_cast_or_null<BroadcastOp>(output.getDefiningOp())) {
            if (this->broadcastOpIsConvBias(broadcast_op)) {
                // Add bias & remap output
                bias = broadcast_op.getInput();
                output = broadcast_op.getInit();
            }
        }

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

        // Get groups
        auto input_type = llvm::dyn_cast_or_null<RankedTensorType>(input.getType());
        if (!input_type || input_type.getRank() != 4) {
            return failure();
        }
        auto groups_attr = IntegerAttr::get(IntegerType::get(this->getContext(), 64), input_type.getDimSize(1));

        // Create the custom convolution with paddings
        rewriter.replaceOpWithNewOp<custom::Conv2DNchwFchwOp>(
            conv_op, resultType, input, weights, bias, output, strides_attr, dilations_attr, paddings_attr, groups_attr
        );

        return success();
    }
};

template<typename PoolingOp>
struct FusePaddingIntoPoolingNchwOp : OpRewritePattern<PoolingOp> {
    using OpRewritePattern<PoolingOp>::OpRewritePattern;

    custom::PoolingMethod getPoolingMethod() const;
    uint64_t getDefaultValue() const;

    LogicalResult matchAndRewrite(PoolingOp pooling_op, PatternRewriter& rewriter) const override {
        // Check and get values
        if (pooling_op.getInputs().size() != 2 || pooling_op.getOutputs().size() != 1 ||
            pooling_op.getResults().size() != 1) {
            return failure();
        }
        Type resultType = pooling_op.getResults()[0].getType();
        Value input = pooling_op.getInputs()[0];
        Value kernel_dummy = pooling_op.getInputs()[1];
        Value output = pooling_op.getOutputs()[0];

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

        // Check that padding is performed with a default value
        auto constant_op = llvm::dyn_cast_or_null<arith::ConstantOp>(yield_op.getValue().getDefiningOp());
        if (!constant_op) {
            return failure();
        }
        if (auto float_attr = llvm::dyn_cast<FloatAttr>(constant_op.getValue())) {
            if (std::bit_cast<uint64_t>(float_attr.getValueAsDouble()) != this->getDefaultValue()) {
                return failure();
            }
        } else if (auto integer_attr = llvm::dyn_cast<IntegerAttr>(constant_op.getValue())) {
            if (integer_attr.getUInt() != this->getDefaultValue()) {
                return failure();
            }
        } else {
            return failure();
        }

        // Get the kernel shape
        auto kernel_dummy_type = llvm::dyn_cast<ShapedType>(kernel_dummy.getType());
        if (kernel_dummy_type.getShape().size() != 2) {
            return failure();
        }
        SmallVector<int64_t> kernel;
        for (int64_t dim : kernel_dummy_type.getShape()) {
            kernel.push_back(dim);
        }

        // Remap the input to the un-padded value
        input = pad_op.getSource();

        // Copy the padding values
        SmallVector<int64_t> paddings;
        paddings.push_back(pad_op.getStaticLow()[2]);
        paddings.push_back(pad_op.getStaticLow()[3]);
        paddings.push_back(pad_op.getStaticHigh()[2]);
        paddings.push_back(pad_op.getStaticHigh()[3]);

        // Copy strides and dilations
        SmallVector<int64_t> strides, dilations;
        for (APInt stride : pooling_op.getStrides()) {
            strides.push_back(stride.getSExtValue());
        }
        for (APInt dilation : pooling_op.getDilations()) {
            dilations.push_back(dilation.getSExtValue());
        }
        auto kernel_attr = DenseI64ArrayAttr::get(this->getContext(), kernel);
        auto strides_attr = DenseI64ArrayAttr::get(this->getContext(), strides);
        auto dilations_attr = DenseI64ArrayAttr::get(this->getContext(), dilations);
        auto paddings_attr = DenseI64ArrayAttr::get(this->getContext(), paddings);

        // Create the custom convolution with paddings
        rewriter.replaceOpWithNewOp<custom::PoolingNchwOp>(
            pooling_op,
            resultType,
            input,
            output,
            kernel_attr,
            strides_attr,
            dilations_attr,
            paddings_attr,
            this->getPoolingMethod()
        );

        return failure();
    }
};

template<>
custom::PoolingMethod FusePaddingIntoPoolingNchwOp<PoolingNchwMaxOp>::getPoolingMethod() const {
    return custom::PoolingMethod::max;
}

template<>
custom::PoolingMethod FusePaddingIntoPoolingNchwOp<PoolingNchwSumOp>::getPoolingMethod() const {
    return custom::PoolingMethod::sum;
}

template<>
uint64_t FusePaddingIntoPoolingNchwOp<PoolingNchwMaxOp>::getDefaultValue() const {
    return 0xFFF0000000000000; // -infinity
}

template<>
uint64_t FusePaddingIntoPoolingNchwOp<PoolingNchwSumOp>::getDefaultValue() const {
    return 0x0000000000000000;
}

struct LinalgCustomFusePaddingPass : public impl::LinalgCustomFusePaddingPassBase<LinalgCustomFusePaddingPass> {
    using LinalgCustomFusePaddingPassBase::LinalgCustomFusePaddingPassBase;

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&this->getContext());
        populateLinalgCustomFusePaddingPass(patterns);
        if (failed(applyPatternsAndFoldGreedily(this->getOperation(), std::move(patterns)))) {
            this->signalPassFailure();
        }
    }
};

void populateLinalgCustomFusePaddingPass(RewritePatternSet& patterns) {
    patterns.add<
        FusePaddingIntoConv2DNchwFchw,
        FusePaddingIntoDepthwiseConv2DNchwChwOp,
        FusePaddingIntoPoolingNchwOp<PoolingNchwMaxOp>,
        FusePaddingIntoPoolingNchwOp<PoolingNchwSumOp>>(patterns.getContext());
}

} // namespace linalg
} // namespace mlir
