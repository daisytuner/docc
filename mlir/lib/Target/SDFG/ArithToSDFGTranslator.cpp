#include "mlir/Target/SDFG/ArithToSDFGTranslator.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "mlir/Target/SDFG/helper.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace mlir {
namespace sdfg {

template<typename Op, ::sdfg::data_flow::TaskletCode code>
LogicalResult translateArithBinaryOp(SDFGTranslator& translator, Op* op) {
    Value lhs = op->getLhs();
    Value rhs = op->getRhs();
    Value result = op->getResult();

    // Types must be primitive for now
    if (!is_sdfg_primitive(lhs.getType()) || !is_sdfg_primitive(rhs.getType()) ||
        !is_sdfg_primitive(result.getType())) {
        return op->emitOpError("Only SDFG primitive types are supported");
    }

    auto& builder = translator.builder();
    auto& block = builder.add_block(translator.insertion_point());
    auto& lhs_access = builder.add_access(block, translator.get_or_create_container(lhs));
    auto& rhs_access = builder.add_access(block, translator.get_or_create_container(rhs));
    auto& result_access = builder.add_access(block, translator.get_or_create_container(result));
    auto& tasklet = builder.add_tasklet(block, code, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, lhs_access, tasklet, "_in1", {});
    builder.add_computational_memlet(block, rhs_access, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", result_access, {});

    return success();
}

template<typename Op>
LogicalResult translateArithCastOp(SDFGTranslator& translator, Op* op) {
    Value in = op->getIn();
    Value result = op->getResult();

    // Types must be primitive for now
    if (!is_sdfg_primitive(in.getType()) || !is_sdfg_primitive(result.getType())) {
        return op->emitOpError("Only SDFG primitive types are supported");
    }

    auto& builder = translator.builder();
    auto& block = builder.add_block(translator.insertion_point());
    auto& in_access = builder.add_access(block, translator.get_or_create_container(in));
    auto& result_access = builder.add_access(block, translator.get_or_create_container(result));
    auto& tasklet = builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, in_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", result_access, {});

    return success();
}

LogicalResult translateArithCmpFOp(SDFGTranslator& translator, arith::CmpFOp* cmpf_op) {
    Value lhs = cmpf_op->getLhs();
    Value rhs = cmpf_op->getRhs();
    Value result = cmpf_op->getResult();

    // Types must be primitive for now
    if (!is_sdfg_primitive(lhs.getType()) || !is_sdfg_primitive(rhs.getType()) ||
        !is_sdfg_primitive(result.getType())) {
        return cmpf_op->emitOpError("Only SDFG primitive types are supported");
    }

    auto& builder = translator.builder();
    auto& block = builder.add_block(translator.insertion_point());

    // Handle AlwaysFalse and AlwaysTrue separately
    if (cmpf_op->getPredicate() == arith::CmpFPredicate::AlwaysFalse ||
        cmpf_op->getPredicate() == arith::CmpFPredicate::AlwaysTrue) {
        std::string val = (cmpf_op->getPredicate() == arith::CmpFPredicate::AlwaysTrue) ? "true" : "false";
        ::sdfg::types::Scalar bool_type(::sdfg::types::PrimitiveType::Bool);

        auto& constant_access = builder.add_constant(block, val, bool_type);
        auto& result_access = builder.add_access(block, translator.get_or_create_container(result));
        auto& tasklet = builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, constant_access, tasklet, "_in", {});
        builder.add_computational_memlet(block, tasklet, "_out", result_access, {});

        return success();
    }

    // Determine supported tasklet code from predicate
    ::sdfg::data_flow::TaskletCode code;
    switch (cmpf_op->getPredicate()) {
        case arith::CmpFPredicate::OEQ:
            code = ::sdfg::data_flow::TaskletCode::fp_oeq;
            break;
        case arith::CmpFPredicate::OGT:
            code = ::sdfg::data_flow::TaskletCode::fp_ogt;
            break;
        case arith::CmpFPredicate::OGE:
            code = ::sdfg::data_flow::TaskletCode::fp_oge;
            break;
        case arith::CmpFPredicate::OLT:
            code = ::sdfg::data_flow::TaskletCode::fp_olt;
            break;
        case arith::CmpFPredicate::OLE:
            code = ::sdfg::data_flow::TaskletCode::fp_ole;
            break;
        case arith::CmpFPredicate::ONE:
            code = ::sdfg::data_flow::TaskletCode::fp_one;
            break;
        case arith::CmpFPredicate::ORD:
            code = ::sdfg::data_flow::TaskletCode::fp_ord;
            break;
        case arith::CmpFPredicate::UEQ:
            code = ::sdfg::data_flow::TaskletCode::fp_ueq;
            break;
        case arith::CmpFPredicate::UGT:
            code = ::sdfg::data_flow::TaskletCode::fp_ugt;
            break;
        case arith::CmpFPredicate::UGE:
            code = ::sdfg::data_flow::TaskletCode::fp_uge;
            break;
        case arith::CmpFPredicate::ULT:
            code = ::sdfg::data_flow::TaskletCode::fp_ult;
            break;
        case arith::CmpFPredicate::ULE:
            code = ::sdfg::data_flow::TaskletCode::fp_ule;
            break;
        case arith::CmpFPredicate::UNE:
            code = ::sdfg::data_flow::TaskletCode::fp_une;
            break;
        case arith::CmpFPredicate::UNO:
            code = ::sdfg::data_flow::TaskletCode::fp_uno;
            break;
        default:
            return failure();
    }

    auto& lhs_access = builder.add_access(block, translator.get_or_create_container(lhs));
    auto& rhs_access = builder.add_access(block, translator.get_or_create_container(rhs));
    auto& result_access = builder.add_access(block, translator.get_or_create_container(result));
    auto& tasklet = builder.add_tasklet(block, code, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, lhs_access, tasklet, "_in1", {});
    builder.add_computational_memlet(block, rhs_access, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", result_access, {});

    return success();
}

LogicalResult translateArithCmpIOp(SDFGTranslator& translator, arith::CmpIOp* cmpi_op) {
    Value lhs = cmpi_op->getLhs();
    Value rhs = cmpi_op->getRhs();
    Value result = cmpi_op->getResult();

    // Types must be primitive for now
    if (!is_sdfg_primitive(lhs.getType()) || !is_sdfg_primitive(rhs.getType()) ||
        !is_sdfg_primitive(result.getType())) {
        return cmpi_op->emitOpError("Only SDFG primitive types are supported");
    }

    // Determine supported tasklet code from predicate
    ::sdfg::data_flow::TaskletCode code;
    switch (cmpi_op->getPredicate()) {
        case arith::CmpIPredicate::eq:
            code = ::sdfg::data_flow::TaskletCode::int_eq;
            break;
        case arith::CmpIPredicate::ne:
            code = ::sdfg::data_flow::TaskletCode::int_ne;
            break;
        case arith::CmpIPredicate::slt:
            code = ::sdfg::data_flow::TaskletCode::int_slt;
            break;
        case arith::CmpIPredicate::sle:
            code = ::sdfg::data_flow::TaskletCode::int_sle;
            break;
        case arith::CmpIPredicate::sgt:
            code = ::sdfg::data_flow::TaskletCode::int_sgt;
            break;
        case arith::CmpIPredicate::sge:
            code = ::sdfg::data_flow::TaskletCode::int_sge;
            break;
        case arith::CmpIPredicate::ult:
            code = ::sdfg::data_flow::TaskletCode::int_ult;
            break;
        case arith::CmpIPredicate::ule:
            code = ::sdfg::data_flow::TaskletCode::int_ule;
            break;
        case arith::CmpIPredicate::ugt:
            code = ::sdfg::data_flow::TaskletCode::int_ugt;
            break;
        case arith::CmpIPredicate::uge:
            code = ::sdfg::data_flow::TaskletCode::int_uge;
            break;
    }

    auto& builder = translator.builder();
    auto& block = builder.add_block(translator.insertion_point());
    auto& lhs_access = builder.add_access(block, translator.get_or_create_container(lhs));
    auto& rhs_access = builder.add_access(block, translator.get_or_create_container(rhs));
    auto& result_access = builder.add_access(block, translator.get_or_create_container(result));
    auto& tasklet = builder.add_tasklet(block, code, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, lhs_access, tasklet, "_in1", {});
    builder.add_computational_memlet(block, rhs_access, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", result_access, {});

    return success();
}

LogicalResult translateArithConstantOp(SDFGTranslator& translator, arith::ConstantOp* constant_op) {
    Value result = constant_op->getResult();

    // Types must be primitive for now
    if (!is_sdfg_primitive(result.getType())) {
        return constant_op->emitOpError("Only SDFG primitive types are supported");
    }

    std::string val = llvm::TypeSwitch<TypedAttr, std::string>(constant_op->getValue())
                          .Case<FloatAttr>([](FloatAttr attr) {
                              return std::to_string(attr.getValue().convertToDouble());
                          })
                          .Case<IntegerAttr>([](IntegerAttr attr) { return std::to_string(attr.getInt()); })
                          .Default([](TypedAttr attr) { return ""; });
    if (val.empty()) {
        return constant_op->emitOpError("Can not convert attribute to constant");
    }
    auto val_type = translator.convertType(constant_op->getValue().getType());
    if (!val_type) {
        return constant_op->emitOpError("Can not convert attribute type");
    }

    auto& builder = translator.builder();
    auto& block = builder.add_block(translator.insertion_point());
    auto& in_access = builder.add_constant(block, val, *val_type);
    auto& result_access = builder.add_access(block, translator.get_or_create_container(result));
    auto& tasklet = builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, in_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", result_access, {});

    return success();
}

LogicalResult translateArithNegFOp(SDFGTranslator& translator, arith::NegFOp* negf_op) {
    Value operand = negf_op->getOperand();
    Value result = negf_op->getResult();

    // Types must be primitive for now
    if (!is_sdfg_primitive(operand.getType()) || !is_sdfg_primitive(result.getType())) {
        return negf_op->emitOpError("Only SDFG primitive types are supported");
    }

    auto& builder = translator.builder();
    auto& block = builder.add_block(translator.insertion_point());
    auto& in_access = builder.add_access(block, translator.get_or_create_container(operand));
    auto& result_access = builder.add_access(block, translator.get_or_create_container(result));
    auto& tasklet = builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::fp_neg, "_out", {"_in"});
    builder.add_computational_memlet(block, in_access, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", result_access, {});

    return success();
}

LogicalResult translateArithOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<arith::AddFOp>([&](arith::AddFOp addf_op) {
            return translateArithBinaryOp<arith::AddFOp, ::sdfg::data_flow::TaskletCode::fp_add>(translator, &addf_op);
        })
        .Case<arith::AddIOp>([&](arith::AddIOp addi_op) {
            return translateArithBinaryOp<arith::AddIOp, ::sdfg::data_flow::TaskletCode::int_add>(translator, &addi_op);
        })
        .Case<arith::AndIOp>([&](arith::AndIOp andi_op) {
            return translateArithBinaryOp<arith::AndIOp, ::sdfg::data_flow::TaskletCode::int_and>(translator, &andi_op);
        })
        .Case<arith::DivFOp>([&](arith::DivFOp divf_op) {
            return translateArithBinaryOp<arith::DivFOp, ::sdfg::data_flow::TaskletCode::fp_div>(translator, &divf_op);
        })
        .Case<arith::DivSIOp>([&](arith::DivSIOp divsi_op) {
            return translateArithBinaryOp<arith::DivSIOp, ::sdfg::data_flow::TaskletCode::int_sdiv>(translator, &divsi_op);
        })
        .Case<arith::DivUIOp>([&](arith::DivUIOp divui_op) {
            return translateArithBinaryOp<arith::DivUIOp, ::sdfg::data_flow::TaskletCode::int_udiv>(translator, &divui_op);
        })
        .Case<arith::MaxSIOp>([&](arith::MaxSIOp maxsi_op) {
            return translateArithBinaryOp<arith::MaxSIOp, ::sdfg::data_flow::TaskletCode::int_smax>(translator, &maxsi_op);
        })
        .Case<arith::MaxUIOp>([&](arith::MaxUIOp maxui_op) {
            return translateArithBinaryOp<arith::MaxUIOp, ::sdfg::data_flow::TaskletCode::int_umax>(translator, &maxui_op);
        })
        .Case<arith::MinSIOp>([&](arith::MinSIOp minsi_op) {
            return translateArithBinaryOp<arith::MinSIOp, ::sdfg::data_flow::TaskletCode::int_smin>(translator, &minsi_op);
        })
        .Case<arith::MinUIOp>([&](arith::MinUIOp minui_op) {
            return translateArithBinaryOp<arith::MinUIOp, ::sdfg::data_flow::TaskletCode::int_umin>(translator, &minui_op);
        })
        .Case<arith::MulFOp>([&](arith::MulFOp mulf_op) {
            return translateArithBinaryOp<arith::MulFOp, ::sdfg::data_flow::TaskletCode::fp_mul>(translator, &mulf_op);
        })
        .Case<arith::MulIOp>([&](arith::MulIOp muli_op) {
            return translateArithBinaryOp<arith::MulIOp, ::sdfg::data_flow::TaskletCode::int_mul>(translator, &muli_op);
        })
        .Case<arith::OrIOp>([&](arith::OrIOp ori_op) {
            return translateArithBinaryOp<arith::OrIOp, ::sdfg::data_flow::TaskletCode::int_or>(translator, &ori_op);
        })
        .Case<arith::RemFOp>([&](arith::RemFOp remf_op) {
            return translateArithBinaryOp<arith::RemFOp, ::sdfg::data_flow::TaskletCode::fp_rem>(translator, &remf_op);
        })
        .Case<arith::RemSIOp>([&](arith::RemSIOp remsi_op) {
            return translateArithBinaryOp<arith::RemSIOp, ::sdfg::data_flow::TaskletCode::int_srem>(translator, &remsi_op);
        })
        .Case<arith::RemUIOp>([&](arith::RemUIOp remui_op) {
            return translateArithBinaryOp<arith::RemUIOp, ::sdfg::data_flow::TaskletCode::int_urem>(translator, &remui_op);
        })
        .Case<arith::ShLIOp>([&](arith::ShLIOp shli_op) {
            return translateArithBinaryOp<arith::ShLIOp, ::sdfg::data_flow::TaskletCode::int_shl>(translator, &shli_op);
        })
        .Case<arith::ShRSIOp>([&](arith::ShRSIOp shrsi_op) {
            return translateArithBinaryOp<arith::ShRSIOp, ::sdfg::data_flow::TaskletCode::int_ashr>(translator, &shrsi_op);
        })
        .Case<arith::ShRUIOp>([&](arith::ShRUIOp shrui_op) {
            return translateArithBinaryOp<arith::ShRUIOp, ::sdfg::data_flow::TaskletCode::int_lshr>(translator, &shrui_op);
        })
        .Case<arith::SubFOp>([&](arith::SubFOp subf_op) {
            return translateArithBinaryOp<arith::SubFOp, ::sdfg::data_flow::TaskletCode::fp_sub>(translator, &subf_op);
        })
        .Case<arith::SubIOp>([&](arith::SubIOp subi_op) {
            return translateArithBinaryOp<arith::SubIOp, ::sdfg::data_flow::TaskletCode::int_sub>(translator, &subi_op);
        })
        .Case<arith::XOrIOp>([&](arith::XOrIOp xori_op) {
            return translateArithBinaryOp<arith::XOrIOp, ::sdfg::data_flow::TaskletCode::int_xor>(translator, &xori_op);
        })
        .Case<arith::BitcastOp>([&](arith::BitcastOp bitcast_op) {
            return translateArithCastOp<arith::BitcastOp>(translator, &bitcast_op);
        })
        .Case<arith::ExtFOp>([&](arith::ExtFOp extf_op) {
            return translateArithCastOp<arith::ExtFOp>(translator, &extf_op);
        })
        .Case<arith::ExtSIOp>([&](arith::ExtSIOp extsi_op) {
            return translateArithCastOp<arith::ExtSIOp>(translator, &extsi_op);
        })
        .Case<arith::ExtUIOp>([&](arith::ExtUIOp extui_op) {
            return translateArithCastOp<arith::ExtUIOp>(translator, &extui_op);
        })
        .Case<arith::FPToSIOp>([&](arith::FPToSIOp fptosi_op) {
            return translateArithCastOp<arith::FPToSIOp>(translator, &fptosi_op);
        })
        .Case<arith::FPToUIOp>([&](arith::FPToUIOp fptoui_op) {
            return translateArithCastOp<arith::FPToUIOp>(translator, &fptoui_op);
        })
        .Case<arith::SIToFPOp>([&](arith::SIToFPOp sitofp_op) {
            return translateArithCastOp<arith::SIToFPOp>(translator, &sitofp_op);
        })
        .Case<arith::TruncFOp>([&](arith::TruncFOp truncf_op) {
            return translateArithCastOp<arith::TruncFOp>(translator, &truncf_op);
        })
        .Case<arith::TruncIOp>([&](arith::TruncIOp trunci_op) {
            return translateArithCastOp<arith::TruncIOp>(translator, &trunci_op);
        })
        .Case<arith::UIToFPOp>([&](arith::UIToFPOp uitofp_op) {
            return translateArithCastOp<arith::UIToFPOp>(translator, &uitofp_op);
        })
        .Case<arith::CmpFOp>([&](arith::CmpFOp cmpf_op) { return translateArithCmpFOp(translator, &cmpf_op); })
        .Case<arith::CmpIOp>([&](arith::CmpIOp cmpi_op) { return translateArithCmpIOp(translator, &cmpi_op); })
        .Case<arith::ConstantOp>([&](arith::ConstantOp constant_op) {
            return translateArithConstantOp(translator, &constant_op);
        })
        .Default([&](Operation* op) { return op->emitError("Unknown operation from arith dialect encountered"); });
}

} // namespace sdfg
} // namespace mlir
