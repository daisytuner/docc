#include <cstddef>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgCustomOps.h"
#include "mlir/Dialect/Linalg/Transforms/CustomTransforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGCUSTOMSPECIALIZEGENERICOPSPASS
#include "mlir/Dialect/Linalg/CustomPasses.h.inc"

namespace linalg {

// TODO Improve regarding constants. Needs explicit hoisting rather than general
struct LinalgGenericConstDependentHoisting : public OpRewritePattern<GenericOp> {
    using OpRewritePattern<GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter) const override {
        Region& region = generic_op.getRegion();
        if (region.getBlocks().size() != 1) {
            return failure();
        }
        Block& block = region.getBlocks().front();
        bool applied;
        do {
            applied = false;
            for (auto& op : block.getOperations()) {
                bool const_dependent = true;
                for (auto operand : op.getOperands()) {
                    if (!llvm::dyn_cast_or_null<arith::ConstantOp>(operand.getDefiningOp())) {
                        const_dependent = false;
                        break;
                    }
                }
                if (const_dependent) {
                    rewriter.moveOpBefore(&op, generic_op);
                    applied = true;
                    break;
                }
            }
        } while (applied);

        return success(applied);
    }
};

// Base class
struct LinalgGenericRewriter : public OpRewritePattern<GenericOp> {
    using OpRewritePattern<GenericOp>::OpRewritePattern;

    bool has_identity_indexing_maps(GenericOp generic_op) const {
        if (generic_op.getIndexingMaps().size() != generic_op.getInputs().size() + generic_op.getOutputs().size()) {
            return false;
        }
        for (size_t i = 0; i < generic_op.getIndexingMaps().size(); i++) {
            auto affine_map_attr = llvm::dyn_cast_or_null<AffineMapAttr>(generic_op.getIndexingMaps()[i]);
            if (!affine_map_attr) {
                return false;
            }
            auto affine_map = affine_map_attr.getAffineMap();
            if (affine_map.isIdentity()) {
                continue;
            }
            Value value = (i < generic_op.getInputs().size())
                              ? generic_op.getInputs()[i]
                              : generic_op.getOutputs()[i - generic_op.getInputs().size()];
            RankedTensorType tensor_type = llvm::dyn_cast_or_null<RankedTensorType>(value.getType());
            if (!tensor_type) {
                return false;
            }
            if (tensor_type.getShape().size() != affine_map.getNumResults()) {
                return false;
            }
            for (size_t i = 0; i < affine_map.getNumResults(); i++) {
                auto expr = affine_map.getResult(i);
                if (auto dim_expr = llvm::dyn_cast<AffineDimExpr>(expr)) {
                    if (dim_expr.getPosition() != i) {
                        return false;
                    }
                } else if (auto constant_expr = llvm::dyn_cast<AffineConstantExpr>(expr)) {
                    if (tensor_type.getShape()[i] != 1 || constant_expr.getValue() != 0) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    bool has_parallel_iterator_types(ArrayAttr iterator_types) const {
        for (auto iterator_type : iterator_types) {
            auto iterator_type_attr = llvm::dyn_cast_or_null<IteratorTypeAttr>(iterator_type);
            if (!iterator_type_attr) {
                return false;
            }
            if (iterator_type_attr.getValue() != utils::IteratorType::parallel) {
                return false;
            }
        }
        return true;
    }

    bool has_elementwise_types(GenericOp generic_op) const {
        // Collect all types
        SmallVector<Type> types;
        for (auto input : generic_op.getInputs()) {
            types.push_back(input.getType());
        }
        for (auto output : generic_op.getOutputs()) {
            types.push_back(output.getType());
        }
        for (auto result : generic_op.getResults()) {
            types.push_back(result.getType());
        }

        // No comparison necessary
        if (types.size() <= 1) {
            return true;
        }

        // First type as comparison
        ShapedType compare_type = llvm::dyn_cast<ShapedType>(types[0]);
        if (!compare_type) {
            return false;
        }
        ArrayRef<int64_t> compare_shape = compare_type.getShape();

        // Compare all types
        for (size_t i = 1; i < types.size(); i++) {
            ShapedType type = llvm::dyn_cast<ShapedType>(types[i]);
            if (!type || failed(verifyCompatibleShape(compare_shape, type.getShape()))) {
                return false;
            }
        }

        return true;
    }

    template<typename Op>
    Op get_single_region_op(Region& region) const {
        if (region.getBlocks().size() != 1) {
            return nullptr;
        }
        auto& block = region.getBlocks().front();
        if (block.getOperations().size() != 2) {
            return nullptr;
        }
        if (!llvm::dyn_cast_or_null<YieldOp>(block.getOperations().back())) {
            return nullptr;
        }
        Op op = llvm::dyn_cast_or_null<Op>(block.getOperations().front());
        return op;
    }

    mlir::FailureOr<SmallVector<Value>>
    convert_inputs(SmallVector<Value> sub_inputs, GenericOp generic_op, PatternRewriter& rewriter) const {
        auto& block = generic_op.getRegion().front();
        for (size_t i = generic_op.getInputs().size(); i < block.getNumArguments(); i++) {
            for (auto sub_input : sub_inputs) {
                if (sub_input == block.getArgument(i)) {
                    return failure();
                }
            }
        }

        SmallVector<Value> inputs;
        for (auto sub_input : sub_inputs) {
            bool found_in_arguments = false;
            size_t i;
            for (i = 0; i < generic_op.getInputs().size(); i++) {
                if (sub_input == block.getArgument(i)) {
                    found_in_arguments = true;
                    break;
                }
            }
            if (found_in_arguments) {
                inputs.push_back(generic_op.getInputs()[i]);
                continue;
            }
            auto constant_op = llvm::dyn_cast_or_null<arith::ConstantOp>(sub_input.getDefiningOp());
            if (!constant_op) {
                return failure();
            }
            if (llvm::dyn_cast_or_null<TensorType>(constant_op.getType())) {
                inputs.push_back(sub_input);
                continue;
            } else if (auto integer_type = llvm::dyn_cast_or_null<IntegerType>(constant_op.getType())) {
                RankedTensorType new_type = RankedTensorType::get({}, integer_type);
                IntegerAttr integer_attr = llvm::dyn_cast<IntegerAttr>(constant_op.getValue());
                DenseIntElementsAttr new_attr =
                    DenseIntElementsAttr::get(new_type, ArrayRef<APInt>{integer_attr.getValue()});
                arith::ConstantOp new_constant_op =
                    rewriter.create<arith::ConstantOp>(constant_op.getLoc(), new_type, new_attr);
                inputs.push_back(new_constant_op);
            } else if (auto float_type = llvm::dyn_cast_or_null<FloatType>(constant_op.getType())) {
                RankedTensorType new_type = RankedTensorType::get({}, float_type);
                FloatAttr float_attr = llvm::dyn_cast<FloatAttr>(constant_op.getValue());
                DenseFPElementsAttr new_attr =
                    DenseFPElementsAttr::get(new_type, ArrayRef<APFloat>{float_attr.getValue()});
                arith::ConstantOp new_constant_op =
                    rewriter.create<arith::ConstantOp>(constant_op.getLoc(), new_type, new_attr);
                inputs.push_back(new_constant_op);
            } else {
                return failure();
            }
        }
        return inputs;
    }

    mlir::FailureOr<SmallVector<Value>> convert_inputs(Value sub_input, GenericOp generic_op, PatternRewriter& rewriter)
        const {
        return this->convert_inputs(SmallVector<Value>({sub_input}), generic_op, rewriter);
    }

    mlir::FailureOr<SmallVector<Value>>
    convert_inputs(Value sub_input1, Value sub_input2, GenericOp generic_op, PatternRewriter& rewriter) const {
        return this->convert_inputs(SmallVector<Value>({sub_input1, sub_input2}), generic_op, rewriter);
    }
};

template<typename ElemOp, typename LinalgElemOp>
struct LinalgGenericToLinalgElementwiseUnary : public LinalgGenericRewriter {
    using LinalgGenericRewriter::LinalgGenericRewriter;

    LogicalResult matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter) const override {
        if (!this->has_identity_indexing_maps(generic_op)) {
            return failure();
        }

        if (!this->has_parallel_iterator_types(generic_op.getIteratorTypes())) {
            return failure();
        }

        if (!this->has_elementwise_types(generic_op)) {
            return failure();
        }

        auto elem_op = this->get_single_region_op<ElemOp>(generic_op.getRegion());
        if (!elem_op) {
            return failure();
        }

        auto inputs = this->convert_inputs(elem_op.getOperand(), generic_op, rewriter);
        if (failed(inputs)) {
            return failure();
        }

        rewriter.replaceOpWithNewOp<LinalgElemOp>(generic_op, *inputs, generic_op.getOutputs());

        return success();
    }
};

template<typename ElemOp, typename LinalgElemOp>
struct LinalgGenericToLinalgElementwiseBinary : public LinalgGenericRewriter {
    using LinalgGenericRewriter::LinalgGenericRewriter;

    LogicalResult matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter) const override {
        if (!this->has_identity_indexing_maps(generic_op)) {
            return failure();
        }

        if (!this->has_parallel_iterator_types(generic_op.getIteratorTypes())) {
            return failure();
        }

        if (!this->has_elementwise_types(generic_op)) {
            return failure();
        }

        auto elem_op = this->get_single_region_op<ElemOp>(generic_op.getRegion());
        if (!elem_op) {
            return failure();
        }

        auto inputs = this->convert_inputs(elem_op.getLhs(), elem_op.getRhs(), generic_op, rewriter);
        if (failed(inputs)) {
            return failure();
        }

        rewriter.replaceOpWithNewOp<LinalgElemOp>(generic_op, *inputs, generic_op.getOutputs());

        return success();
    }
};

struct LinalgGenericToSpecialLinalgDivF : public LinalgGenericRewriter {
    using LinalgGenericRewriter::LinalgGenericRewriter;

    LogicalResult matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter) const override {
        if (!this->has_identity_indexing_maps(generic_op)) {
            return failure();
        }

        if (!this->has_parallel_iterator_types(generic_op.getIteratorTypes())) {
            return failure();
        }

        if (!this->has_elementwise_types(generic_op)) {
            return failure();
        }

        Region& region = generic_op.getRegion();
        if (region.getBlocks().size() != 1) {
            return failure();
        }
        auto& block = region.getBlocks().front();
        if (block.getOperations().size() != 4) {
            return failure();
        }
        auto ops_0 = block.getOperations().begin();
        auto ops_1 = std::next(ops_0);
        auto ops_2 = std::next(ops_1);
        auto ops_3 = std::next(ops_2);
        if (!(block.getNumArguments() == 2 || block.getNumArguments() == 3)) {
            return failure();
        }

        arith::DivFOp divf_op = llvm::dyn_cast_or_null<arith::DivFOp>(*ops_2);
        if (!divf_op) {
            return failure();
        }
        Value dividend = divf_op.getLhs();
        Value divisor = divf_op.getRhs();

        arith::CmpFOp cmpf_op = llvm::dyn_cast_or_null<arith::CmpFOp>(*ops_0);
        if (!cmpf_op || cmpf_op.getPredicate() != arith::CmpFPredicate::ONE || cmpf_op.getLhs() != divisor) {
            return failure();
        }
        arith::ConstantOp constant_op = llvm::dyn_cast_or_null<arith::ConstantOp>(cmpf_op.getRhs().getDefiningOp());
        if (!constant_op) {
            return failure();
        }
        FloatAttr float_attr = llvm::dyn_cast_or_null<FloatAttr>(constant_op.getValue());
        if (!float_attr || float_attr.getValueAsDouble() != 0.0) {
            return failure();
        }

        cf::AssertOp assert_op = llvm::dyn_cast_or_null<cf::AssertOp>(*ops_1);
        if (!assert_op || assert_op.getArg() != cmpf_op) {
            return failure();
        }

        YieldOp yield_op = llvm::dyn_cast_or_null<YieldOp>(*ops_3);
        if (!yield_op || yield_op.getValues().size() != 1 || yield_op.getValues()[0] != divf_op) {
            return failure();
        }

        auto inputs = this->convert_inputs(dividend, divisor, generic_op, rewriter);
        if (failed(inputs)) {
            return failure();
        }

        rewriter.replaceOpWithNewOp<DivOp>(generic_op, *inputs, generic_op.getOutputs());

        return success();
    }
};

struct LinalgGenericToLinalgCustomReLU : public LinalgGenericRewriter {
    using LinalgGenericRewriter::LinalgGenericRewriter;

    LogicalResult matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter) const override {
        if (!this->has_identity_indexing_maps(generic_op)) {
            return failure();
        }

        if (!this->has_parallel_iterator_types(generic_op.getIteratorTypes())) {
            return failure();
        }

        if (!this->has_elementwise_types(generic_op)) {
            return failure();
        }

        Region& region = generic_op.getRegion();
        if (region.getBlocks().size() != 1) {
            return failure();
        }
        auto& block = region.getBlocks().front();
        if (block.getOperations().size() != 3) {
            return failure();
        }
        auto ops_0 = block.getOperations().begin();
        auto ops_1 = std::next(ops_0);
        auto ops_2 = std::next(ops_1);
        if (block.getNumArguments() != 2) {
            return failure();
        }

        arith::CmpFOp cmpf_op = llvm::dyn_cast_or_null<arith::CmpFOp>(*ops_0);
        if (!cmpf_op || cmpf_op.getPredicate() != arith::CmpFPredicate::UGT ||
            cmpf_op.getLhs() != block.getArgument(0)) {
            return failure();
        }
        arith::ConstantOp constant_op = llvm::dyn_cast_or_null<arith::ConstantOp>(cmpf_op.getRhs().getDefiningOp());
        if (!constant_op) {
            return failure();
        }
        FloatAttr float_attr = llvm::dyn_cast_or_null<FloatAttr>(constant_op.getValue());
        if (!float_attr || float_attr.getValueAsDouble() != 0.0) {
            return failure();
        }

        arith::SelectOp select_op = llvm::dyn_cast_or_null<arith::SelectOp>(*ops_1);
        if (!select_op || select_op.getCondition() != cmpf_op || select_op.getTrueValue() != block.getArgument(0) ||
            select_op.getFalseValue() != constant_op) {
            return failure();
        }

        YieldOp yield_op = llvm::dyn_cast_or_null<YieldOp>(*ops_2);
        if (!yield_op || yield_op.getValues().size() != 1 || yield_op.getValues()[0] != select_op) {
            return failure();
        }

        auto inputs = this->convert_inputs(select_op.getTrueValue(), generic_op, rewriter);
        if (failed(inputs) || inputs->size() != 1 || generic_op.getNumResults() != 1) {
            return failure();
        }

        rewriter.replaceOpWithNewOp<
            custom::ReLUOp>(generic_op, generic_op.getResult(0).getType(), (*inputs)[0], generic_op.getOutputs()[0]);

        return success();
    }
};

struct LinalgGenericToLinalgCustomSigmoid : public LinalgGenericRewriter {
    using LinalgGenericRewriter::LinalgGenericRewriter;

    LogicalResult matchAndRewrite(GenericOp generic_op, PatternRewriter& rewriter) const override {
        if (!this->has_identity_indexing_maps(generic_op)) {
            return failure();
        }

        if (!this->has_parallel_iterator_types(generic_op.getIteratorTypes())) {
            return failure();
        }

        if (!this->has_elementwise_types(generic_op)) {
            return failure();
        }

        Region& region = generic_op.getRegion();
        if (region.getBlocks().size() != 1) {
            return failure();
        }
        auto& block = region.getBlocks().front();
        if (block.getOperations().size() != 5) {
            return failure();
        }
        auto ops_0 = block.getOperations().begin();
        auto ops_1 = std::next(ops_0);
        auto ops_2 = std::next(ops_1);
        auto ops_3 = std::next(ops_2);
        auto ops_4 = std::next(ops_3);
        if (block.getNumArguments() != 2) {
            return failure();
        }

        arith::NegFOp negf_op = llvm::dyn_cast_or_null<arith::NegFOp>(*ops_0);
        if (!negf_op || negf_op.getOperand() != block.getArgument(0)) {
            return failure();
        }

        math::ExpOp exp_op = llvm::dyn_cast_or_null<math::ExpOp>(*ops_1);
        if (!exp_op || exp_op.getOperand() != negf_op) {
            return failure();
        }

        arith::AddFOp addf_op = llvm::dyn_cast_or_null<arith::AddFOp>(*ops_2);
        if (!addf_op) {
            return failure();
        }
        Value addf_const_value;
        if (addf_op.getLhs() == exp_op) {
            addf_const_value = addf_op.getRhs();
        } else if (addf_op.getRhs() == exp_op) {
            addf_const_value = addf_op.getLhs();
        } else {
            return failure();
        }
        arith::ConstantOp constant_op = llvm::dyn_cast_or_null<arith::ConstantOp>(addf_const_value.getDefiningOp());
        if (!constant_op) {
            return failure();
        }
        FloatAttr float_attr = llvm::dyn_cast_or_null<FloatAttr>(constant_op.getValue());
        if (!float_attr || float_attr.getValueAsDouble() != 1.0) {
            return failure();
        }

        arith::DivFOp divf_op = llvm::dyn_cast_or_null<arith::DivFOp>(*ops_3);
        if (!divf_op || divf_op.getLhs() != constant_op || divf_op.getRhs() != addf_op) {
            return failure();
        }

        YieldOp yield_op = llvm::dyn_cast_or_null<YieldOp>(*ops_4);
        if (!yield_op || yield_op.getValues().size() != 1 || yield_op.getValues()[0] != divf_op) {
            return failure();
        }

        auto inputs = this->convert_inputs(negf_op.getOperand(), generic_op, rewriter);
        if (failed(inputs) || inputs->size() != 1 || generic_op.getNumResults() != 1) {
            return failure();
        }

        rewriter.replaceOpWithNewOp<
            custom::SigmoidOp>(generic_op, generic_op.getResult(0).getType(), (*inputs)[0], generic_op.getOutputs()[0]);

        return success();
    }
};

struct LinalgCustomSpecializeGenericOpsPass
    : public impl::LinalgCustomSpecializeGenericOpsPassBase<LinalgCustomSpecializeGenericOpsPass> {
    using LinalgCustomSpecializeGenericOpsPassBase::LinalgCustomSpecializeGenericOpsPassBase;

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&this->getContext());
        populateLinalgCustomSpecializeGenericOpsPass(patterns);
        if (failed(applyPatternsAndFoldGreedily(this->getOperation(), std::move(patterns)))) {
            this->signalPassFailure();
        }
    }
};

void populateLinalgCustomSpecializeGenericOpsPass(RewritePatternSet& patterns) {
    patterns.add<
        // LinalgGenericConstDependentHoisting, // Decativated for now because it also hoists linalg.index
        LinalgGenericToLinalgElementwiseUnary<math::ExpOp, ExpOp>,
        LinalgGenericToLinalgElementwiseUnary<math::SqrtOp, SqrtOp>,
        LinalgGenericToLinalgElementwiseBinary<arith::AddFOp, AddOp>,
        LinalgGenericToLinalgElementwiseBinary<arith::AddIOp, AddOp>,
        LinalgGenericToLinalgElementwiseBinary<arith::DivFOp, DivOp>,
        LinalgGenericToLinalgElementwiseBinary<arith::DivSIOp, DivOp>,
        LinalgGenericToLinalgElementwiseBinary<arith::DivUIOp, DivOp>,
        LinalgGenericToLinalgElementwiseBinary<arith::MulFOp, MulOp>,
        LinalgGenericToLinalgElementwiseBinary<arith::MulIOp, MulOp>,
        LinalgGenericToLinalgElementwiseBinary<arith::SubFOp, SubOp>,
        LinalgGenericToLinalgElementwiseBinary<arith::SubIOp, SubOp>,
        LinalgGenericToSpecialLinalgDivF,
        LinalgGenericToLinalgCustomReLU,
        LinalgGenericToLinalgCustomSigmoid>(patterns.getContext());
}

} // namespace linalg
} // namespace mlir
