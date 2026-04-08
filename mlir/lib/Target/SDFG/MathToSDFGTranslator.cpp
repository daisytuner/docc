#include "mlir/Target/SDFG/MathToSDFGTranslator.h"

#include <llvm/ADT/TypeSwitch.h>

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "mlir/Target/SDFG/helper.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/tasklet_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"

namespace mlir {
namespace sdfg {

template<typename Op, ::sdfg::math::cmath::CMathFunction function>
LogicalResult translateMathUnaryOp(SDFGTranslator& translator, Op* op) {
    Value operand = op->getOperand();
    Value result = op->getResult();
    auto deb_info = translator.get_debug_info(op->getOperationName(), op->getLoc());

    auto& builder = translator.builder();
    auto operand_container = translator.get_or_create_container(operand);
    auto result_container = translator.get_or_create_container(result);

    if (is_sdfg_primitive(operand.getType()) && is_sdfg_primitive(result.getType())) {
        auto& operand_container_type = builder.subject().type(operand_container);
        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& operand_access = builder.add_access(block, operand_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& libnode = builder.add_library_node<
            ::sdfg::math::cmath::CMathNode>(block, deb_info, function, operand_container_type.primitive_type());
        builder.add_computational_memlet(block, operand_access, libnode, "_in1", {}, operand_container_type, deb_info);
        builder.add_computational_memlet(
            block, libnode, "_out", result_access, {}, builder.subject().type(result_container), deb_info
        );
    } else if (is_tensor_of_sdfg_primitive(operand.getType()) && is_tensor_of_sdfg_primitive(result.getType())) {
        auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
        auto tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

        auto element_type = translator.convertType(result_tensor_type.getElementType());
        auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

        uint64_t size = 1;
        for (int64_t dim : tensor_info.shape()) {
            size *= dim;
        }
        translator.handle_malloc(
            result_container,
            ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*element_type)),
            deb_info
        );

        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& operand_access = builder.add_access(block, operand_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& libnode = builder.add_library_node<::sdfg::math::tensor::CMathTensorNode>(
            block,
            deb_info,
            function,
            std::vector<std::string>({"_out"}),
            std::vector<std::string>({"_in"}),
            sdfg_tensor->shape()
        );
        builder.add_computational_memlet(block, operand_access, libnode, "_in", {}, *sdfg_tensor, deb_info);
        builder.add_computational_memlet(block, libnode, "_out", result_access, {}, *sdfg_tensor, deb_info);
    } else {
        return op->emitOpError("Unsupported type(s)");
    }

    return success();
}

template<typename Op, ::sdfg::math::cmath::CMathFunction function>
LogicalResult translateMathBinaryOp(SDFGTranslator& translator, Op* op) {
    Value lhs = op->getLhs();
    Value rhs = op->getRhs();
    Value result = op->getResult();
    auto deb_info = translator.get_debug_info(op->getOperationName(), op->getLoc());

    auto& builder = translator.builder();
    auto lhs_container = translator.get_or_create_container(lhs);
    auto rhs_container = translator.get_or_create_container(rhs);
    auto result_container = translator.get_or_create_container(result);

    if (is_sdfg_primitive(lhs.getType()) && is_sdfg_primitive(rhs.getType()) && is_sdfg_primitive(result.getType())) {
        auto& lhs_container_type = builder.subject().type(lhs_container);
        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& lhs_access = builder.add_access(block, lhs_container, deb_info);
        auto& rhs_access = builder.add_access(block, rhs_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& libnode = builder.add_library_node<
            ::sdfg::math::cmath::CMathNode>(block, deb_info, function, lhs_container_type.primitive_type());
        builder.add_computational_memlet(block, lhs_access, libnode, "_in1", {}, lhs_container_type, deb_info);
        builder.add_computational_memlet(
            block, rhs_access, libnode, "_in2", {}, builder.subject().type(rhs_container), deb_info
        );
        builder.add_computational_memlet(
            block, libnode, "_out", result_access, {}, builder.subject().type(result_container), deb_info
        );
    } else if (is_tensor_of_sdfg_primitive(lhs.getType()) && is_tensor_of_sdfg_primitive(lhs.getType()) &&
               is_tensor_of_sdfg_primitive(result.getType())) {
        auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
        auto tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

        auto element_type = translator.convertType(result_tensor_type.getElementType());
        auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

        uint64_t size = 1;
        for (int64_t dim : tensor_info.shape()) {
            size *= dim;
        }
        translator.handle_malloc(
            result_container,
            ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*element_type)),
            deb_info
        );

        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& lhs_access = builder.add_access(block, lhs_container, deb_info);
        auto& rhs_access = builder.add_access(block, rhs_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& libnode = builder.add_library_node<::sdfg::math::tensor::CMathTensorNode>(
            block,
            deb_info,
            function,
            std::vector<std::string>({"_out"}),
            std::vector<std::string>({"_in1", "_in2"}),
            sdfg_tensor->shape()
        );
        builder.add_computational_memlet(block, lhs_access, libnode, "_in1", {}, *sdfg_tensor, deb_info);
        builder.add_computational_memlet(block, rhs_access, libnode, "_in2", {}, *sdfg_tensor, deb_info);
        builder.add_computational_memlet(block, libnode, "_out", result_access, {}, *sdfg_tensor, deb_info);
    } else {
        return op->emitOpError("Unsupported type(s)");
    }

    return success();
}

LogicalResult translateMathAbsIOp(SDFGTranslator& translator, math::AbsIOp* absi_op) {
    Value operand = absi_op->getOperand();
    Value result = absi_op->getResult();
    auto deb_info = translator.get_debug_info(absi_op->getOperationName(), absi_op->getLoc());

    auto& builder = translator.builder();
    auto operand_container = translator.get_or_create_container(operand);
    auto result_container = translator.get_or_create_container(result);

    if (is_sdfg_primitive(operand.getType()) && is_sdfg_primitive(result.getType())) {
        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& operand_access = builder.add_access(block, operand_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& tasklet = builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::int_abs, "_out", {"_in"}, deb_info);
        builder.add_computational_memlet(block, operand_access, tasklet, "_in", {}, deb_info);
        builder.add_computational_memlet(block, tasklet, "_out", result_access, {}, deb_info);
    } else if (is_tensor_of_sdfg_primitive(operand.getType()) && is_tensor_of_sdfg_primitive(result.getType())) {
        auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
        auto tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

        auto element_type = translator.convertType(result_tensor_type.getElementType());
        auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

        uint64_t size = 1;
        for (int64_t dim : tensor_info.shape()) {
            size *= dim;
        }
        translator.handle_malloc(
            result_container,
            ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*element_type)),
            deb_info
        );

        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& operand_access = builder.add_access(block, operand_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& libnode = builder.add_library_node<::sdfg::math::tensor::TaskletTensorNode>(
            block,
            deb_info,
            ::sdfg::data_flow::TaskletCode::int_abs,
            std::vector<std::string>({"_out"}),
            std::vector<std::string>({"_in"}),
            sdfg_tensor->shape()
        );
        builder.add_computational_memlet(block, operand_access, libnode, "_in", {}, *sdfg_tensor, deb_info);
        builder.add_computational_memlet(block, libnode, "_out", result_access, {}, *sdfg_tensor, deb_info);
    } else {
        return absi_op->emitOpError("Unsupported type(s)");
    }

    return success();
}

LogicalResult translateMathFmaOp(SDFGTranslator& translator, math::FmaOp* fma_op) {
    Value a = fma_op->getA();
    Value b = fma_op->getB();
    Value c = fma_op->getC();
    Value result = fma_op->getResult();
    auto deb_info = translator.get_debug_info(fma_op->getOperationName(), fma_op->getLoc());

    auto& builder = translator.builder();
    auto a_container = translator.get_or_create_container(a);
    auto b_container = translator.get_or_create_container(b);
    auto c_container = translator.get_or_create_container(c);
    auto result_container = translator.get_or_create_container(result);

    if (is_sdfg_primitive(a.getType()) && is_sdfg_primitive(b.getType()) && is_sdfg_primitive(c.getType()) &&
        is_sdfg_primitive(result.getType())) {
        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& a_access = builder.add_access(block, a_container, deb_info);
        auto& b_access = builder.add_access(block, b_container, deb_info);
        auto& c_access = builder.add_access(block, c_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& tasklet =
            builder
                .add_tasklet(block, ::sdfg::data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"}, deb_info);
        builder.add_computational_memlet(block, a_access, tasklet, "_in1", {}, deb_info);
        builder.add_computational_memlet(block, b_access, tasklet, "_in2", {}, deb_info);
        builder.add_computational_memlet(block, c_access, tasklet, "_in3", {}, deb_info);
        builder.add_computational_memlet(block, tasklet, "_out", result_access, {}, deb_info);
    } else if (is_tensor_of_sdfg_primitive(a.getType()) && is_tensor_of_sdfg_primitive(b.getType()) &&
               is_tensor_of_sdfg_primitive(c.getType()) && is_tensor_of_sdfg_primitive(result.getType())) {
        auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
        auto tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

        auto element_type = translator.convertType(result_tensor_type.getElementType());
        auto sdfg_tensor = tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*element_type));

        uint64_t size = 1;
        for (int64_t dim : tensor_info.shape()) {
            size *= dim;
        }
        translator.handle_malloc(
            result_container,
            ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*element_type)),
            deb_info
        );

        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& a_access = builder.add_access(block, a_container, deb_info);
        auto& b_access = builder.add_access(block, b_container, deb_info);
        auto& c_access = builder.add_access(block, c_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& libnode = builder.add_library_node<::sdfg::math::tensor::TaskletTensorNode>(
            block,
            deb_info,
            ::sdfg::data_flow::TaskletCode::fp_fma,
            std::vector<std::string>({"_out"}),
            std::vector<std::string>({"_in1", "_in2", "_in3"}),
            sdfg_tensor->shape()
        );
        builder.add_computational_memlet(block, a_access, libnode, "_in1", {}, *sdfg_tensor, deb_info);
        builder.add_computational_memlet(block, b_access, libnode, "_in2", {}, *sdfg_tensor, deb_info);
        builder.add_computational_memlet(block, c_access, libnode, "_in3", {}, *sdfg_tensor, deb_info);
        builder.add_computational_memlet(block, libnode, "_out", result_access, {}, *sdfg_tensor, deb_info);
    } else {
        return fma_op->emitOpError("Unsupported type(s)");
    }

    return success();
}

LogicalResult translateMathFPowIOp(SDFGTranslator& translator, math::FPowIOp* fpowi_op) {
    Value lhs = fpowi_op->getLhs();
    Value rhs = fpowi_op->getRhs();
    Value result = fpowi_op->getResult();
    auto deb_info = translator.get_debug_info(fpowi_op->getOperationName(), fpowi_op->getLoc());

    auto& builder = translator.builder();
    auto lhs_container = translator.get_or_create_container(lhs);
    auto rhs_container = translator.get_or_create_container(rhs);
    auto result_container = translator.get_or_create_container(result);

    if (is_sdfg_primitive(lhs.getType()) && is_sdfg_primitive(rhs.getType()) && is_sdfg_primitive(result.getType())) {
        auto& lhs_container_type = builder.subject().type(lhs_container);
        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& lhs_access = builder.add_access(block, lhs_container, deb_info);
        auto& rhs_access = builder.add_access(block, rhs_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& libnode = builder.add_library_node<::sdfg::math::cmath::CMathNode>(
            block, deb_info, ::sdfg::math::cmath::CMathFunction::pow, lhs_container_type.primitive_type()
        );
        builder.add_computational_memlet(block, lhs_access, libnode, "_in1", {}, lhs_container_type, deb_info);
        builder.add_computational_memlet(block, rhs_access, libnode, "_in2", {}, lhs_container_type, deb_info);
        builder.add_computational_memlet(
            block, libnode, "_out", result_access, {}, builder.subject().type(result_container), deb_info
        );
    } else if (is_tensor_of_sdfg_primitive(lhs.getType()) && is_tensor_of_sdfg_primitive(lhs.getType()) &&
               is_tensor_of_sdfg_primitive(result.getType())) {
        auto result_tensor_type = llvm::dyn_cast<TensorType>(result.getType());
        auto result_tensor_info = translator.get_or_create_tensor_info(result_container, result_tensor_type);

        auto lhs_tensor_type = llvm::dyn_cast<TensorType>(lhs.getType());
        auto lhs_tensor_info = translator.get_or_create_tensor_info(lhs_container, lhs_tensor_type);

        auto rhs_tensor_type = llvm::dyn_cast<TensorType>(rhs.getType());
        auto rhs_tensor_info = translator.get_or_create_tensor_info(rhs_container, rhs_tensor_type);

        auto result_element_type = translator.convertType(result_tensor_type.getElementType());
        auto result_sdfg_tensor =
            result_tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*result_element_type));

        auto lhs_element_type = translator.convertType(lhs_tensor_type.getElementType());
        auto lhs_sdfg_tensor = lhs_tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*lhs_element_type));

        auto rhs_element_type = translator.convertType(rhs_tensor_type.getElementType());
        auto rhs_sdfg_tensor = rhs_tensor_info.get_sdfg_tensor(static_cast<::sdfg::types::Scalar&>(*rhs_element_type));

        uint64_t size = 1;
        for (int64_t dim : result_tensor_info.shape()) {
            size *= dim;
        }
        translator.handle_malloc(
            result_container,
            ::sdfg::symbolic::mul(::sdfg::symbolic::integer(size), ::sdfg::symbolic::size_of_type(*result_element_type)),
            deb_info
        );

        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& lhs_access = builder.add_access(block, lhs_container, deb_info);
        auto& rhs_access = builder.add_access(block, rhs_container, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& libnode = builder.add_library_node<::sdfg::math::tensor::CMathTensorNode>(
            block,
            deb_info,
            ::sdfg::math::cmath::CMathFunction::pow,
            std::vector<std::string>({"_out"}),
            std::vector<std::string>({"_in1", "_in2"}),
            result_sdfg_tensor->shape()
        );
        builder.add_computational_memlet(block, lhs_access, libnode, "_in1", {}, *lhs_sdfg_tensor, deb_info);
        builder.add_computational_memlet(block, rhs_access, libnode, "_in2", {}, *rhs_sdfg_tensor, deb_info);
        builder.add_computational_memlet(block, libnode, "_out", result_access, {}, *result_sdfg_tensor, deb_info);
    } else {
        return fpowi_op->emitOpError("Unsupported type(s)");
    }

    return success();
}

LogicalResult translateMathRsqrtOp(SDFGTranslator& translator, math::RsqrtOp* rsqrt_op) {
    Value operand = rsqrt_op->getOperand();
    Value result = rsqrt_op->getResult();
    auto deb_info = translator.get_debug_info(rsqrt_op->getOperationName(), rsqrt_op->getLoc());

    auto& builder = translator.builder();
    auto operand_container = translator.get_or_create_container(operand);
    auto result_container = translator.get_or_create_container(result);
    auto tmp_container = builder.find_new_name();
    builder.add_container(tmp_container, *translator.convertType(result.getType()));

    if (is_sdfg_primitive(operand.getType()) && is_sdfg_primitive(result.getType())) {
        auto& operand_container_type = builder.subject().type(operand_container);
        auto& result_container_Type = builder.subject().type(result_container);
        auto& block = builder.add_block(translator.insertion_point(), {}, deb_info);
        auto& operand_access = builder.add_access(block, operand_container, deb_info);
        auto& tmp_access = builder.add_access(block, tmp_container, deb_info);
        auto& one_constant = builder.add_constant(block, "1.0", operand_container_type, deb_info);
        auto& result_access = builder.add_access(block, result_container, deb_info);
        auto& libnode = builder.add_library_node<::sdfg::math::cmath::CMathNode>(
            block, deb_info, ::sdfg::math::cmath::CMathFunction::sqrt, operand_container_type.primitive_type()
        );
        builder.add_computational_memlet(block, operand_access, libnode, "_in1", {}, operand_container_type, deb_info);
        builder.add_computational_memlet(block, libnode, "_out", tmp_access, {}, result_container_Type, deb_info);
        auto& tasklet =
            builder.add_tasklet(block, ::sdfg::data_flow::TaskletCode::fp_div, "_out", {"_in1", "_in2"}, deb_info);
        builder.add_computational_memlet(block, one_constant, tasklet, "_in1", {}, deb_info);
        builder.add_computational_memlet(block, tmp_access, tasklet, "_in2", {}, deb_info);
        builder.add_computational_memlet(block, tasklet, "_out", result_access, {}, deb_info);
    } else {
        return rsqrt_op->emitOpError("Unsupported type(s)");
    }

    return success();
}

LogicalResult translateMathOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<math::AbsFOp>([&](math::AbsFOp absf_op) {
            return translateMathUnaryOp<math::AbsFOp, ::sdfg::math::cmath::CMathFunction::fabs>(translator, &absf_op);
        })
        .Case<math::AbsIOp>([&](math::AbsIOp absi_op) { return translateMathAbsIOp(translator, &absi_op); })
        .Case<math::AcosOp>([&](math::AcosOp acos_op) {
            return translateMathUnaryOp<math::AcosOp, ::sdfg::math::cmath::CMathFunction::acos>(translator, &acos_op);
        })
        .Case<math::AcoshOp>([&](math::AcoshOp acosh_op) {
            return translateMathUnaryOp<math::AcoshOp, ::sdfg::math::cmath::CMathFunction::acosh>(translator, &acosh_op);
        })
        .Case<math::AsinOp>([&](math::AsinOp asin_op) {
            return translateMathUnaryOp<math::AsinOp, ::sdfg::math::cmath::CMathFunction::asin>(translator, &asin_op);
        })
        .Case<math::AsinhOp>([&](math::AsinhOp asinh_op) {
            return translateMathUnaryOp<math::AsinhOp, ::sdfg::math::cmath::CMathFunction::asinh>(translator, &asinh_op);
        })
        .Case<math::AtanOp>([&](math::AtanOp atan_op) {
            return translateMathUnaryOp<math::AtanOp, ::sdfg::math::cmath::CMathFunction::atan>(translator, &atan_op);
        })
        .Case<math::Atan2Op>([&](math::Atan2Op atan2_op) {
            return translateMathBinaryOp<math::Atan2Op, ::sdfg::math::cmath::CMathFunction::atan2>(translator, &atan2_op);
        })
        .Case<math::AtanhOp>([&](math::AtanhOp atanh_op) {
            return translateMathUnaryOp<math::AtanhOp, ::sdfg::math::cmath::CMathFunction::atanh>(translator, &atanh_op);
        })
        .Case<math::CbrtOp>([&](math::CbrtOp cbrt_op) {
            return translateMathUnaryOp<math::CbrtOp, ::sdfg::math::cmath::CMathFunction::cbrt>(translator, &cbrt_op);
        })
        .Case<math::CeilOp>([&](math::CeilOp ceil_op) {
            return translateMathUnaryOp<math::CeilOp, ::sdfg::math::cmath::CMathFunction::ceil>(translator, &ceil_op);
        })
        // Missing ClampFOp
        .Case<math::CopySignOp>([&](math::CopySignOp copysign_op) {
            return translateMathBinaryOp<
                math::CopySignOp,
                ::sdfg::math::cmath::CMathFunction::copysign>(translator, &copysign_op);
        })
        .Case<math::CosOp>([&](math::CosOp cos_op) {
            return translateMathUnaryOp<math::CosOp, ::sdfg::math::cmath::CMathFunction::cos>(translator, &cos_op);
        })
        .Case<math::CoshOp>([&](math::CoshOp cosh_op) {
            return translateMathUnaryOp<math::CoshOp, ::sdfg::math::cmath::CMathFunction::cosh>(translator, &cosh_op);
        })
        // Missing CountLeadingZerosOp
        // Missing CtPopOp
        // Missing CountTrailingZerosOp
        .Case<math::ErfOp>([&](math::ErfOp erf_op) {
            return translateMathUnaryOp<math::ErfOp, ::sdfg::math::cmath::CMathFunction::erf>(translator, &erf_op);
        })
        .Case<math::ExpOp>([&](math::ExpOp exp_op) {
            return translateMathUnaryOp<math::ExpOp, ::sdfg::math::cmath::CMathFunction::exp>(translator, &exp_op);
        })
        .Case<math::Exp2Op>([&](math::Exp2Op exp2_op) {
            return translateMathUnaryOp<math::Exp2Op, ::sdfg::math::cmath::CMathFunction::exp2>(translator, &exp2_op);
        })
        .Case<math::ExpM1Op>([&](math::ExpM1Op expm1_op) {
            return translateMathUnaryOp<math::ExpM1Op, ::sdfg::math::cmath::CMathFunction::expm1>(translator, &expm1_op);
        })
        .Case<math::FloorOp>([&](math::FloorOp floor_op) {
            return translateMathUnaryOp<math::FloorOp, ::sdfg::math::cmath::CMathFunction::floor>(translator, &floor_op);
        })
        .Case<math::FmaOp>([&](math::FmaOp fma_op) { return translateMathFmaOp(translator, &fma_op); })
        .Case<math::FPowIOp>([&](math::FPowIOp fpowi_op) { return translateMathFPowIOp(translator, &fpowi_op); })
        // Missing IPowIOp
        // Missing IsFiniteOp
        // Missing IsInfOp
        // Missing IsNanOp
        // Missing IsNormalOp
        .Case<math::LogOp>([&](math::LogOp log_op) {
            return translateMathUnaryOp<math::LogOp, ::sdfg::math::cmath::CMathFunction::log>(translator, &log_op);
        })
        .Case<math::Log10Op>([&](math::Log10Op log10_op) {
            return translateMathUnaryOp<math::Log10Op, ::sdfg::math::cmath::CMathFunction::log10>(translator, &log10_op);
        })
        .Case<math::Log1pOp>([&](math::Log1pOp log1p_op) {
            return translateMathUnaryOp<math::Log1pOp, ::sdfg::math::cmath::CMathFunction::log1p>(translator, &log1p_op);
        })
        .Case<math::Log2Op>([&](math::Log2Op log2_op) {
            return translateMathUnaryOp<math::Log2Op, ::sdfg::math::cmath::CMathFunction::log2>(translator, &log2_op);
        })
        .Case<math::PowFOp>([&](math::PowFOp powf_op) {
            return translateMathBinaryOp<math::PowFOp, ::sdfg::math::cmath::CMathFunction::pow>(translator, &powf_op);
        })
        .Case<math::RoundOp>([&](math::RoundOp round_op) {
            return translateMathUnaryOp<math::RoundOp, ::sdfg::math::cmath::CMathFunction::round>(translator, &round_op);
        })
        .Case<math::RoundEvenOp>([&](math::RoundEvenOp roundeven_op) {
            return translateMathUnaryOp<
                math::RoundEvenOp,
                ::sdfg::math::cmath::CMathFunction::roundeven>(translator, &roundeven_op);
        })
        .Case<math::RsqrtOp>([&](math::RsqrtOp rsqrt_op) { return translateMathRsqrtOp(translator, &rsqrt_op); })
        .Case<math::SinOp>([&](math::SinOp sin_op) {
            return translateMathUnaryOp<math::SinOp, ::sdfg::math::cmath::CMathFunction::sin>(translator, &sin_op);
        })
        // Missing SincosOp
        .Case<math::SinhOp>([&](math::SinhOp sinh_op) {
            return translateMathUnaryOp<math::SinhOp, ::sdfg::math::cmath::CMathFunction::sinh>(translator, &sinh_op);
        })
        .Case<math::SqrtOp>([&](math::SqrtOp sqrt_op) {
            return translateMathUnaryOp<math::SqrtOp, ::sdfg::math::cmath::CMathFunction::sqrt>(translator, &sqrt_op);
        })
        .Case<math::TanOp>([&](math::TanOp tan_op) {
            return translateMathUnaryOp<math::TanOp, ::sdfg::math::cmath::CMathFunction::tan>(translator, &tan_op);
        })
        .Case<math::TanhOp>([&](math::TanhOp tanh_op) {
            return translateMathUnaryOp<math::TanhOp, ::sdfg::math::cmath::CMathFunction::tanh>(translator, &tanh_op);
        })
        .Case<math::TruncOp>([&](math::TruncOp trunc_op) {
            return translateMathUnaryOp<math::TruncOp, ::sdfg::math::cmath::CMathFunction::trunc>(translator, &trunc_op);
        })
        .Default([&](Operation* op) {
            return op->emitError("Unknown operation from math dialect encountered: ") << op->getName();
        });
}

} // namespace sdfg
} // namespace mlir
