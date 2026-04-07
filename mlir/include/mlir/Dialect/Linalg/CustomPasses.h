#pragma once

#include "mlir/Dialect/Linalg/Transforms/CustomTransforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

#define GEN_PASS_DECL
#include "mlir/Dialect/Linalg/CustomPasses.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Linalg/CustomPasses.h.inc"

} // namespace mlir
