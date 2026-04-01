#include "mlir/Dialect/Linalg/IR/LinalgCustomOps.h"
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
