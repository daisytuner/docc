#pragma once

#include <llvm-19/llvm/Support/LogicalResult.h>
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"

namespace mlir {
namespace sdfg {

LogicalResult translateLinalgOp(SDFGTranslator& translator, Operation* op);

LogicalResult translateLinalgFillOp(SDFGTranslator& translator, linalg::FillOp* op);

LogicalResult translateLinalgMatmulOp(SDFGTranslator& translator, linalg::MatmulOp* op);

LogicalResult translateLinalgTransposeOp(SDFGTranslator& translator, linalg::TransposeOp* op);

LogicalResult translateLinalgBroadcastOp(SDFGTranslator& translator, linalg::BroadcastOp* op);

LogicalResult translateLinalgConv2DNchwFchwOp(SDFGTranslator& translator, linalg::Conv2DNchwFchwOp* op);

} // namespace sdfg
} // namespace mlir
