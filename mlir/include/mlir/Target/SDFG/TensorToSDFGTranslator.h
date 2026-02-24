#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"

namespace mlir {
namespace sdfg {

LogicalResult translateTensorOp(SDFGTranslator& translator, Operation* op);

} // namespace sdfg
} // namespace mlir
