#include "mlir/Conversion/ArithToSDFG/ArithToSDFG.h"

#include <llvm/Support/LogicalResult.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

#define GEN_PASS_DEF_CONVERTARITHTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace arith2sdfg {

template<typename OrigOp, sdfg::TaskletCode code>
struct BinaryOpConversion : public OpRewritePattern<OrigOp> {
    using OpRewritePattern<OrigOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(OrigOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // Types must be primitive for now
        if (!sdfg::is_primitive(op.getLhs().getType()) || !sdfg::is_primitive(op.getRhs().getType()) ||
            !sdfg::is_primitive(op.getResult().getType())) {
            return failure();
        }

        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Type result_type = op.getResult().getType();

        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result_type}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        sdfg::MemletOp lhs_memlet_op = rewriter.create<sdfg::MemletOp>(block_op.getLoc(), lhs.getType(), lhs);
        sdfg::MemletOp rhs_memlet_op = rewriter.create<sdfg::MemletOp>(lhs_memlet_op.getLoc(), rhs.getType(), rhs);
        sdfg::TaskletOp tasklet_op = rewriter.create<sdfg::TaskletOp>(
            rhs_memlet_op.getLoc(), result_type, code, SmallVector<Value>({lhs_memlet_op, rhs_memlet_op})
        );
        sdfg::MemletOp result_memlet_op = rewriter.create<sdfg::MemletOp>(tasklet_op.getLoc(), result_type, tasklet_op);

        block_op.getBody().front().back().setOperands(SmallVector<Value>({result_memlet_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

typedef BinaryOpConversion<arith::AddFOp, sdfg::TaskletCode::fp_add> AddFOpConversion;
typedef BinaryOpConversion<arith::AddIOp, sdfg::TaskletCode::int_add> AddIOpConversion;
typedef BinaryOpConversion<arith::AndIOp, sdfg::TaskletCode::int_and> AndIOpConversion;
typedef BinaryOpConversion<arith::DivFOp, sdfg::TaskletCode::fp_div> DivFOpConversion;
typedef BinaryOpConversion<arith::DivSIOp, sdfg::TaskletCode::int_sdiv> DivSIOpConversion;
typedef BinaryOpConversion<arith::DivUIOp, sdfg::TaskletCode::int_udiv> DivUIOpConversion;
typedef BinaryOpConversion<arith::MaxSIOp, sdfg::TaskletCode::int_smax> MaxSIOpConversion;
typedef BinaryOpConversion<arith::MaxUIOp, sdfg::TaskletCode::int_umax> MaxUIOpConversion;
typedef BinaryOpConversion<arith::MinSIOp, sdfg::TaskletCode::int_smin> MinSIOpConversion;
typedef BinaryOpConversion<arith::MinUIOp, sdfg::TaskletCode::int_umin> MinUIOpConversion;
typedef BinaryOpConversion<arith::MulFOp, sdfg::TaskletCode::fp_mul> MulFOpConversion;
typedef BinaryOpConversion<arith::MulIOp, sdfg::TaskletCode::int_mul> MulIOpConversion;
typedef BinaryOpConversion<arith::OrIOp, sdfg::TaskletCode::int_or> OrIOpConversion;
typedef BinaryOpConversion<arith::RemFOp, sdfg::TaskletCode::fp_rem> RemFOpConversion;
typedef BinaryOpConversion<arith::RemSIOp, sdfg::TaskletCode::int_srem> RemSIOpConversion;
typedef BinaryOpConversion<arith::RemUIOp, sdfg::TaskletCode::int_urem> RemUIOpConversion;
typedef BinaryOpConversion<arith::ShLIOp, sdfg::TaskletCode::int_shl> ShLIOpConversion;
typedef BinaryOpConversion<arith::ShRSIOp, sdfg::TaskletCode::int_ashr> ShRSIOpConversion;
typedef BinaryOpConversion<arith::ShRUIOp, sdfg::TaskletCode::int_lshr> ShRUIOpConversion;
typedef BinaryOpConversion<arith::SubFOp, sdfg::TaskletCode::fp_sub> SubFOpConversion;
typedef BinaryOpConversion<arith::SubIOp, sdfg::TaskletCode::int_sub> SubIOpConversion;
typedef BinaryOpConversion<arith::XOrIOp, sdfg::TaskletCode::int_xor> XOrIOpConversion;

template<typename OrigOp>
struct CastOpConversion : public OpRewritePattern<OrigOp> {
    using OpRewritePattern<OrigOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(OrigOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // Types must be primitive for now
        if (!sdfg::is_primitive(op.getIn().getType()) || !sdfg::is_primitive(op.getResult().getType())) {
            return failure();
        }

        Value in = op.getIn();
        Type result_type = op.getResult().getType();

        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result_type}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        sdfg::MemletOp in_memlet_op = rewriter.create<sdfg::MemletOp>(block_op.getLoc(), in.getType(), in);
        sdfg::TaskletOp tasklet_op = rewriter.create<sdfg::TaskletOp>(
            in_memlet_op.getLoc(), result_type, sdfg::TaskletCode::assign, SmallVector<Value>({in_memlet_op})
        );
        sdfg::MemletOp result_memlet_op = rewriter.create<sdfg::MemletOp>(tasklet_op.getLoc(), result_type, tasklet_op);

        block_op.getBody().front().back().setOperands(SmallVector<Value>({result_memlet_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

typedef CastOpConversion<arith::BitcastOp> BitcastOpConversion;
typedef CastOpConversion<arith::ExtFOp> ExtFOpConversion;
typedef CastOpConversion<arith::ExtSIOp> ExtSIOpConversion;
typedef CastOpConversion<arith::ExtUIOp> ExtUIOpConversion;
typedef CastOpConversion<arith::FPToSIOp> FPToSIOpConversion;
typedef CastOpConversion<arith::FPToUIOp> FPToUIOpConversion;
typedef CastOpConversion<arith::SIToFPOp> SIToFPOpConversion;
typedef CastOpConversion<arith::TruncFOp> TruncFOpConversion;
typedef CastOpConversion<arith::TruncIOp> TruncIOpConversion;
typedef CastOpConversion<arith::UIToFPOp> UIToFPOpConversion;

struct CmpFOpConversion : public OpRewritePattern<arith::CmpFOp> {
    using OpRewritePattern<arith::CmpFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::CmpFOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // Types must be primitive for now
        if (!sdfg::is_primitive(op.getLhs().getType()) || !sdfg::is_primitive(op.getRhs().getType()) ||
            !sdfg::is_primitive(op.getResult().getType())) {
            return failure();
        }

        // Handle AlwaysFalse and AlwaysTrue separately
        if (op.getPredicate() == arith::CmpFPredicate::AlwaysFalse ||
            op.getPredicate() == arith::CmpFPredicate::AlwaysTrue) {
            Type result_type = op.getResult().getType();

            sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result_type}));
            rewriter.setInsertionPointToStart(&block_op.getBody().front());

            bool val = op.getPredicate() == arith::CmpFPredicate::AlwaysTrue;
            sdfg::ConstantOp constant_op =
                rewriter
                    .create<sdfg::ConstantOp>(block_op.getLoc(), result_type, BoolAttr::get(this->getContext(), val));
            sdfg::TaskletOp tasklet_op = rewriter.create<sdfg::TaskletOp>(
                constant_op.getLoc(), result_type, sdfg::TaskletCode::assign, SmallVector<Value>({constant_op})
            );
            sdfg::MemletOp result_memlet_op =
                rewriter.create<sdfg::MemletOp>(tasklet_op.getLoc(), result_type, tasklet_op);

            block_op.getBody().front().back().setOperands(SmallVector<Value>({result_memlet_op}));
            rewriter.replaceOp(op, block_op);

            return success();
        }

        // Determine supported tasklet code from predicate
        sdfg::TaskletCode code;
        switch (op.getPredicate()) {
            case arith::CmpFPredicate::OEQ:
                code = sdfg::TaskletCode::fp_oeq;
                break;
            case arith::CmpFPredicate::OGT:
                code = sdfg::TaskletCode::fp_ogt;
                break;
            case arith::CmpFPredicate::OGE:
                code = sdfg::TaskletCode::fp_oge;
                break;
            case arith::CmpFPredicate::OLT:
                code = sdfg::TaskletCode::fp_olt;
                break;
            case arith::CmpFPredicate::OLE:
                code = sdfg::TaskletCode::fp_ole;
                break;
            case arith::CmpFPredicate::ONE:
                code = sdfg::TaskletCode::fp_one;
                break;
            case arith::CmpFPredicate::ORD:
                code = sdfg::TaskletCode::fp_ord;
                break;
            case arith::CmpFPredicate::UEQ:
                code = sdfg::TaskletCode::fp_ueq;
                break;
            case arith::CmpFPredicate::UGT:
                code = sdfg::TaskletCode::fp_ugt;
                break;
            case arith::CmpFPredicate::UGE:
                code = sdfg::TaskletCode::fp_uge;
                break;
            case arith::CmpFPredicate::ULT:
                code = sdfg::TaskletCode::fp_ult;
                break;
            case arith::CmpFPredicate::ULE:
                code = sdfg::TaskletCode::fp_ule;
                break;
            case arith::CmpFPredicate::UNE:
                code = sdfg::TaskletCode::fp_une;
                break;
            case arith::CmpFPredicate::UNO:
                code = sdfg::TaskletCode::fp_uno;
                break;
            default:
                return failure();
        }

        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Type result_type = op.getResult().getType();

        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result_type}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        sdfg::MemletOp lhs_memlet_op = rewriter.create<sdfg::MemletOp>(block_op.getLoc(), lhs.getType(), lhs);
        sdfg::MemletOp rhs_memlet_op = rewriter.create<sdfg::MemletOp>(lhs_memlet_op.getLoc(), rhs.getType(), rhs);
        sdfg::TaskletOp tasklet_op = rewriter.create<sdfg::TaskletOp>(
            rhs_memlet_op.getLoc(), result_type, code, SmallVector<Value>({lhs_memlet_op, rhs_memlet_op})
        );
        sdfg::MemletOp result_memlet_op = rewriter.create<sdfg::MemletOp>(tasklet_op.getLoc(), result_type, tasklet_op);

        block_op.getBody().front().back().setOperands(SmallVector<Value>({result_memlet_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

struct CmpIOpConversion : public OpRewritePattern<arith::CmpIOp> {
    using OpRewritePattern<arith::CmpIOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::CmpIOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // Types must be primitive for now
        if (!sdfg::is_primitive(op.getLhs().getType()) || !sdfg::is_primitive(op.getRhs().getType()) ||
            !sdfg::is_primitive(op.getResult().getType())) {
            return failure();
        }

        // Determine supported tasklet code from predicate
        sdfg::TaskletCode code;
        switch (op.getPredicate()) {
            case arith::CmpIPredicate::eq:
                code = sdfg::TaskletCode::int_eq;
                break;
            case arith::CmpIPredicate::ne:
                code = sdfg::TaskletCode::int_ne;
                break;
            case arith::CmpIPredicate::slt:
                code = sdfg::TaskletCode::int_slt;
                break;
            case arith::CmpIPredicate::sle:
                code = sdfg::TaskletCode::int_sle;
                break;
            case arith::CmpIPredicate::sgt:
                code = sdfg::TaskletCode::int_sgt;
                break;
            case arith::CmpIPredicate::sge:
                code = sdfg::TaskletCode::int_sge;
                break;
            case arith::CmpIPredicate::ult:
                code = sdfg::TaskletCode::int_ult;
                break;
            case arith::CmpIPredicate::ule:
                code = sdfg::TaskletCode::int_ule;
                break;
            case arith::CmpIPredicate::ugt:
                code = sdfg::TaskletCode::int_ugt;
                break;
            case arith::CmpIPredicate::uge:
                code = sdfg::TaskletCode::int_uge;
                break;
        }

        Value lhs = op.getLhs();
        Value rhs = op.getRhs();
        Type result_type = op.getResult().getType();

        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result_type}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        sdfg::MemletOp lhs_memlet_op = rewriter.create<sdfg::MemletOp>(block_op.getLoc(), lhs.getType(), lhs);
        sdfg::MemletOp rhs_memlet_op = rewriter.create<sdfg::MemletOp>(lhs_memlet_op.getLoc(), rhs.getType(), rhs);
        sdfg::TaskletOp tasklet_op = rewriter.create<sdfg::TaskletOp>(
            rhs_memlet_op.getLoc(), result_type, code, SmallVector<Value>({lhs_memlet_op, rhs_memlet_op})
        );
        sdfg::MemletOp result_memlet_op = rewriter.create<sdfg::MemletOp>(tasklet_op.getLoc(), result_type, tasklet_op);

        block_op.getBody().front().back().setOperands(SmallVector<Value>({result_memlet_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

struct ConstantOpConversion : public OpRewritePattern<arith::ConstantOp> {
    using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::ConstantOp op, PatternRewriter& rewriter) const override {
        Type type = op.getType();
        if (!sdfg::is_tensor_of_primitive(type)) {
            return failure();
        }
        if (auto elements = dyn_cast<ElementsAttr>(op.getValue())) {
            rewriter.replaceOpWithNewOp<sdfg::TensorConstantOp>(op, type, elements);
            return success();
        }
        return failure();
    }
};

struct NegFOpConversion : public OpRewritePattern<arith::NegFOp> {
    using OpRewritePattern<arith::NegFOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::NegFOp op, PatternRewriter& rewriter) const override {
        // Check that parent is from sdfg dialect
        if (op->getParentOp()->getDialect()->getNamespace() != sdfg::SDFGDialect::getDialectNamespace()) {
            return failure();
        }

        // Types must be primitive for now
        if (!sdfg::is_primitive(op.getOperand().getType()) || !sdfg::is_primitive(op.getResult().getType())) {
            return failure();
        }

        Value operand = op.getOperand();
        Type result_type = op.getResult().getType();

        sdfg::BlockOp block_op = rewriter.create<sdfg::BlockOp>(op.getLoc(), SmallVector<Type>({result_type}));
        rewriter.setInsertionPointToStart(&block_op.getBody().front());

        sdfg::MemletOp operand_memlet_op = rewriter.create<sdfg::MemletOp>(block_op.getLoc(), operand.getType(), operand);
        sdfg::TaskletOp tasklet_op = rewriter.create<sdfg::TaskletOp>(
            operand_memlet_op.getLoc(), result_type, sdfg::TaskletCode::fp_neg, SmallVector<Value>({operand_memlet_op})
        );
        sdfg::MemletOp result_memlet_op = rewriter.create<sdfg::MemletOp>(tasklet_op.getLoc(), result_type, tasklet_op);

        block_op.getBody().front().back().setOperands(SmallVector<Value>({result_memlet_op}));
        rewriter.replaceOp(op, block_op);

        return success();
    }
};

} // namespace arith2sdfg

namespace {

struct ConvertArithToSDFG : public impl::ConvertArithToSDFGBase<ConvertArithToSDFG> {
    using ConvertArithToSDFGBase::ConvertArithToSDFGBase;

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&this->getContext());
        arith::populateArithToSDFGPatterns(patterns);
        if (failed(applyPatternsAndFoldGreedily(this->getOperation(), std::move(patterns)))) {
            this->signalPassFailure();
        }
    }
};

} // namespace

namespace arith {

void populateArithToSDFGPatterns(RewritePatternSet& patterns) {
    patterns.add<
        arith2sdfg::AddFOpConversion, // arith.addf (arith::AddFOp)
        arith2sdfg::AddIOpConversion, // arith.addi (arith::AddIOp)
        // arith.addui_extended (arith::AddUIExtendedOp)
        arith2sdfg::AndIOpConversion, // arith.andi (arith::AndIOp)
        arith2sdfg::BitcastOpConversion, // arith.bitcast (arith::BitcastOp)
        // arith.ceildivsi (arith::CeilDivSIOp)
        // arith.ceildivui (arith::CeilDivUIOp)
        arith2sdfg::CmpFOpConversion, // arith.cmpf (arith::CmpFOp)
        arith2sdfg::CmpIOpConversion, // arith.cmpi (arith::CmpIOp)
        arith2sdfg::ConstantOpConversion, // arith.constant (arith::ConstantOp)
        arith2sdfg::DivFOpConversion, // arith.divf (arith::DivFOp)
        arith2sdfg::DivSIOpConversion, // arith.divsi (arith::DivSIOp)
        arith2sdfg::DivUIOpConversion, // arith.divui (arith::DivUIOp)
        arith2sdfg::ExtFOpConversion, // arith.extf (arith::ExtFOp)
        arith2sdfg::ExtSIOpConversion, // arith.extsi (arith::ExtSIOp)
        arith2sdfg::ExtUIOpConversion, // arith.extui (arith::ExtUIOp)
        // arith.floordivsi (arith::FloorDivSIOp)
        arith2sdfg::FPToSIOpConversion, // arith.fptosi (arith::FPToSIOp)
        arith2sdfg::FPToUIOpConversion, // arith.fptoui (arith::FPToUIOp)
        // arith.index_cast (arith::IndexCastOp)
        // arith.index_castui (arith::IndexCastUIOp)
        // arith.maximumf (arith::MaximumFOp)
        // arith.maxnumf (arith::MaxNumFOp)
        arith2sdfg::MaxSIOpConversion, // arith.maxsi (arith::MaxSIOp)
        arith2sdfg::MaxUIOpConversion, // arith.maxui (arith::MaxUIOp)
        // arith.minimumf (arith::MinimumFOp)
        // arith.minnumf (arith::MinNumFOp)
        arith2sdfg::MinSIOpConversion, // arith.minsi (arith::MinSIOp)
        arith2sdfg::MinUIOpConversion, // arith.minui (arith::MinUIOp)
        arith2sdfg::MulFOpConversion, // arith.mulf (arith::MulFOp)
        arith2sdfg::MulIOpConversion, // arith.muli (arith::MulIOp)
        // arith.mulsi_extended (arith::MulSIExtendedOp)
        // arith.mului_extended (arith::MulUIExtendedOp)
        arith2sdfg::NegFOpConversion, // arith.negf (arith::NegFOp)
        arith2sdfg::OrIOpConversion, // arith.ori (arith::OrIOp)
        arith2sdfg::RemFOpConversion, // arith.remf (arith::RemFOp)
        arith2sdfg::RemSIOpConversion, // arith.remsi (arith::RemSIOp)
        arith2sdfg::RemUIOpConversion, // arith.remui (arith::RemUIOp)
        // arith.scaling_extf (arith::ScalingExtFOp)
        // arith.scaling_truncf (arith::ScalingTruncFOp)
        // arith.select (arith::SelectOp)
        arith2sdfg::ShLIOpConversion, // arith.shli (arith::ShLIOp)
        arith2sdfg::ShRSIOpConversion, // arith.shrsi (arith::ShRSIOp)
        arith2sdfg::ShRUIOpConversion, // arith.shrui (arith::ShRUIOp)
        arith2sdfg::SIToFPOpConversion, // arith.sitofp (arith::SIToFPOp)
        arith2sdfg::SubFOpConversion, // arith.subf (arith::SubFOp)
        arith2sdfg::SubIOpConversion, // arith.subi (arith::SubIOp)
        arith2sdfg::TruncFOpConversion, // arith.truncf (arith::TruncFOp)
        arith2sdfg::TruncIOpConversion, // arith.trunci (arith::TruncIOp)
        arith2sdfg::UIToFPOpConversion, // arith.uitofp (arith::UIToFPOp)
        arith2sdfg::XOrIOpConversion // arith.xori (arith::XOrIOp)
        >(patterns.getContext());
}

} // namespace arith
} // namespace mlir
