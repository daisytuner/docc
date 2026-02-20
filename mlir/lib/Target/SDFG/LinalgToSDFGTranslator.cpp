#include "mlir/Target/SDFG/LinalgToSDFGTranslator.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/LogicalResult.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

namespace mlir {
namespace sdfg {


LogicalResult translateLinalgOp(SDFGTranslator& translator, Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<linalg::FillOp>([&](linalg::FillOp fill_op) { return translateLinalgFillOp(translator, &fill_op); })
        .Case<linalg::MatmulOp>([&](linalg::MatmulOp matmul_op) {
            return translateLinalgMatmulOp(translator, &matmul_op);
        })
        .Case<linalg::TransposeOp>([&](linalg::TransposeOp transpose_op) {
            return translateLinalgTransposeOp(translator, &transpose_op);
        })
        .Default([&](Operation* op) { return op->emitError("Unknown operation from linalg dialect encountered"); });
}

LogicalResult translateLinalgFillOp(SDFGTranslator& translator, linalg::FillOp* op) {
    auto& sequence = translator.insertion_point();

    auto& block = translator.builder().add_block(sequence);

    Value value = op->value();
    Value result = op->result();

    // TODO: add fill node

    if (auto constant_op = dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
        auto& inaccess = translator.builder().add_constant(
            block, translator.convertTypedAttr(constant_op.getValue()), *translator.convertType(constant_op.getType())
        );
        // translator.builder().add_computational_memlet(block, inaccess, )
    } else {
    }

    auto& out_access = translator.builder().add_access(block, translator.get_or_create_container(result));
}

LogicalResult translateLinalgMatmulOp(SDFGTranslator& translator, linalg::MatmulOp* op) {
    auto& sequence = translator.insertion_point();

    auto& block = translator.builder().add_block(sequence);

    // For now, only handle 2D matmul with no transposes or broadcasts
    auto lhs_type = dyn_cast_or_null<RankedTensorType>(op->getOperand(0).getType());
    auto rhs_type = dyn_cast_or_null<RankedTensorType>(op->getOperand(1).getType());
    auto output_type = dyn_cast_or_null<RankedTensorType>(op->getResult(0).getType());
    if (!lhs_type || !rhs_type || !output_type || lhs_type.getRank() != 2 || rhs_type.getRank() != 2 ||
        output_type.getRank() != 2) {
        return op->emitError("Only 2D matmul is supported for now");
    }

    auto in_container_lhs = translator.get_or_create_container(op->getOperand(0));
    auto in_container_rhs = translator.get_or_create_container(op->getOperand(1));
    auto out_container = translator.get_or_create_container(op->getResult(0));

    auto& lhs_access = translator.builder().add_access(block, in_container_lhs);
    auto& rhs_access = translator.builder().add_access(block, in_container_rhs);
    auto& out_access = translator.builder().add_access(block, out_container);

    auto& alpha = translator.builder().add_constant(block, "1.0", *translator.convertType(lhs_type.getElementType()));
    auto& beta = translator.builder().add_constant(block, "0.0", *translator.convertType(output_type.getElementType()));

    auto& tensor_info_lhs = translator.get_or_create_tensor_info(in_container_lhs, lhs_type);
    auto& tensor_info_rhs = translator.get_or_create_tensor_info(in_container_rhs, rhs_type);
    auto& tensor_info_out = translator.get_or_create_tensor_info(out_container, output_type);

    // check if offsets are 0 for all tensors since we don't support partial tensors for now
    if (tensor_info_lhs.offset() != 0 || tensor_info_rhs.offset() != 0 || tensor_info_out.offset() != 0) {
        return op->emitError("Only tensors with 0 offset are supported for now");
    }

    auto m = ::sdfg::symbolic::integer(tensor_info_lhs.shape().at(0));
    auto n = ::sdfg::symbolic::integer(tensor_info_rhs.shape().at(1));
    auto k = ::sdfg::symbolic::integer(tensor_info_lhs.shape().at(1));

    auto lda = ::sdfg::symbolic::integer(tensor_info_lhs.strides().at(0));
    auto ldb = ::sdfg::symbolic::integer(tensor_info_rhs.strides().at(0));
    auto ldc = ::sdfg::symbolic::integer(tensor_info_out.strides().at(0));

    ::sdfg::math::blas::BLAS_Precision precision;
    if (output_type.getElementType().isF16()) {
        precision = ::sdfg::math::blas::BLAS_Precision::h;
    } else if (output_type.getElementType().isF32()) {
        precision = ::sdfg::math::blas::BLAS_Precision::s;
    } else if (output_type.getElementType().isF64()) {
        precision = ::sdfg::math::blas::BLAS_Precision::d;
    } else {
        op->emitOpError("has unsupported element type. Only f16, f32, and f64 are supported.");
        return failure();
    }

    auto& libnode = translator.builder().add_library_node<::sdfg::math::blas::GEMMNode>(
        block,
        ::sdfg::DebugInfo(),
        ::sdfg::math::blas::ImplementationType_BLAS,
        precision,
        ::sdfg::math::blas::BLAS_Layout::RowMajor,
        ::sdfg::math::blas::BLAS_Transpose::No,
        ::sdfg::math::blas::BLAS_Transpose::No,
        m,
        n,
        k,
        lda,
        ldb,
        ldc
    );

    translator.builder()
        .add_computational_memlet(block, lhs_access, libnode, "__A", {}, *translator.convertType(lhs_type));
    translator.builder()
        .add_computational_memlet(block, rhs_access, libnode, "__B", {}, *translator.convertType(rhs_type));
    translator.builder()
        .add_computational_memlet(block, out_access, libnode, "__C", {}, *translator.convertType(output_type));
    translator.builder().add_computational_memlet(block, alpha, libnode, "__alpha", {}, alpha.type());
    translator.builder().add_computational_memlet(block, beta, libnode, "__beta", {}, beta.type());

    auto& write_access = translator.builder().add_access(block, out_container);

    translator.builder()
        .add_computational_memlet(block, libnode, "__C", write_access, {}, *translator.convertType(output_type));

    return success();
}

LogicalResult translateLinalgTransposeOp(SDFGTranslator& translator, linalg::TransposeOp* op) {
    auto& sequence = translator.insertion_point();

    auto& block = translator.builder().add_block(sequence);

    Value input = op->getInput();
    Value result = op->getResult()[0];

    // Check that input and output types are ranked tensors
    auto input_tensor_type = dyn_cast_or_null<TensorType>(input.getType());
    auto result_tensor_type = dyn_cast_or_null<TensorType>(result.getType());
    if (!input_tensor_type || !result_tensor_type) {
        return op->emitError("Input and output types must be ranked tensors");
    }

    auto permutation = op->getPermutation();

    auto in_container = translator.get_or_create_container(input);
    auto out_container = translator.get_or_create_container(result);

    auto& in_access = translator.builder().add_access(block, in_container);
    auto& out_access = translator.builder().add_access(block, out_container);

    translator.builder()
        .add_reference_memlet(block, in_access, out_access, {}, *translator.convertType(input_tensor_type));

    // Compute and store tensor info for input and output tensors. This will be used for libnode generation later on.
    auto& in_tensor_info = translator.get_or_create_tensor_info(in_container, input_tensor_type);

    auto out_tensor_info = in_tensor_info.transpose(permutation);
    translator.tensor_info_map().insert({out_container, out_tensor_info});

    return success();
}

} // namespace sdfg
} // namespace mlir
