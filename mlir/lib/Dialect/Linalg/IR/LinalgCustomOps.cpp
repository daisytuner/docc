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
    RankedTensorType batchType = this->getBatch().getType();
    RankedTensorType eType = this->getE().getType();
    RankedTensorType varType = this->getVar().getType();
    RankedTensorType gammaType = this->getGamma().getType();
    RankedTensorType betaType = this->getBeta().getType();
    RankedTensorType outputType = this->getOutput().getType();

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
// Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//

LogicalResult Conv2DNchwFchwOp::verify() {
    RankedTensorType inputType = this->getInput().getType();
    RankedTensorType weightsType = this->getWeights().getType();
    RankedTensorType outputType = this->getOutput().getType();

    if (inputType.getDimSize(0) != outputType.getDimSize(0)) {
        return this->emitOpError("incompatible N shape");
    }
    if (inputType.getDimSize(1) != weightsType.getDimSize(1)) {
        return this->emitOpError("incompatible C shape");
    }
    if (weightsType.getDimSize(0) != outputType.getDimSize(1)) {
        return this->emitOpError("incompatible F shape");
    }
    if (this->getBias()) {
        RankedTensorType biasType = this->getBias().getType();
        if (weightsType.getDimSize(0) != biasType.getDimSize(0)) {
            return this->emitOpError("incompatible F shape");
        }
    }

    if (this->getStrides().size() != 2) {
        return this->emitOpError("must have exactly two stride values");
    }
    if (this->getDilations().size() != 2) {
        return this->emitOpError("must have exactly two dialtion values");
    }
    if (this->getPaddings().size() != 4) {
        return this->emitOpError("must have exactly four padding values");
    }

    int64_t Ho = (inputType.getDimSize(2) + this->getPaddings()[0] + this->getPaddings()[2] -
                  this->getDilations()[0] * (weightsType.getDimSize(2) - 1) - 1) /
                     this->getStrides()[0] +
                 1;
    int64_t Wo = (inputType.getDimSize(3) + this->getPaddings()[1] + this->getPaddings()[3] -
                  this->getDilations()[1] * (weightsType.getDimSize(3) - 1) - 1) /
                     this->getStrides()[1] +
                 1;
    if (Ho != outputType.getDimSize(2)) {
        return this->emitOpError("output height should be ") << Ho;
    }
    if (Wo != outputType.getDimSize(3)) {
        return this->emitOpError("output width should be ") << Wo;
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
