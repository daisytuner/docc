#include "mlir/Dialect/Linalg/IR/LinalgCustomOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/Linalg/IR/LinalgCustomOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgCustomOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgCustomOps.cpp.inc"

namespace mlir {
namespace linalg {
namespace custom {

//===----------------------------------------------------------------------===//
// ReLUOp
//===----------------------------------------------------------------------===//

LogicalResult ReLUOp::verify() {
    ShapedType inputType = llvm::cast<ShapedType>(this->getInput().getType());
    ShapedType outputType = llvm::cast<ShapedType>(this->getOutput().getType());

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (failed(verifyCompatibleShape(inputShape, outputShape))) {
        return this->emitOpError("incompatible output shape");
    }

    return success();
}

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//

LogicalResult SigmoidOp::verify() {
    ShapedType inputType = llvm::cast<ShapedType>(this->getInput().getType());
    ShapedType outputType = llvm::cast<ShapedType>(this->getOutput().getType());

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (failed(verifyCompatibleShape(inputShape, outputShape))) {
        return this->emitOpError("incompatible output shape");
    }

    return success();
}

//===----------------------------------------------------------------------===//
// BatchNorm2DNchw
//===----------------------------------------------------------------------===//

LogicalResult BatchNorm2DNchwOp::verify() {
    RankedTensorType batchType = llvm::cast<RankedTensorType>(this->getBatch().getType());
    RankedTensorType eType = llvm::cast<RankedTensorType>(this->getE().getType());
    RankedTensorType varType = llvm::cast<RankedTensorType>(this->getVar().getType());
    RankedTensorType gammaType = llvm::cast<RankedTensorType>(this->getGamma().getType());
    RankedTensorType betaType = llvm::cast<RankedTensorType>(this->getBeta().getType());
    RankedTensorType outputType = llvm::cast<RankedTensorType>(this->getOutput().getType());

    ArrayRef<int64_t> batchShape = batchType.getShape();
    ArrayRef<int64_t> eShape = eType.getShape();
    ArrayRef<int64_t> varShape = varType.getShape();
    ArrayRef<int64_t> gammaShape = gammaType.getShape();
    ArrayRef<int64_t> betaShape = betaType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    if (failed(verifyCompatibleShape(batchShape, outputShape))) {
        return this->emitOpError("incompatible batch / output shape");
    }

    int64_t c = batchShape[1];
    if (eShape[0] != c) {
        return this->emitOpError("incompatible e shape");
    }
    if (varShape[0] != c) {
        return this->emitOpError("incompatible var shape");
    }
    if (gammaShape[0] != c) {
        return this->emitOpError("incompatible gamma shape");
    }
    if (betaShape[0] != c) {
        return this->emitOpError("incompatible beta shape");
    }

    return success();
}

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

void LinalgCustomDialect::initialize() {
    this->addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgCustomOps.cpp.inc"
        >();
}

} // namespace custom
} // namespace linalg
} // namespace mlir
