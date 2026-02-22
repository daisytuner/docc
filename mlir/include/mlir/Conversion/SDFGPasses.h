#pragma once

#include "mlir/Conversion/ArithToSDFG/ArithToSDFG.h"
#include "mlir/Conversion/ConvertToSDFG/ConvertToSDFG.h"
#include "mlir/Conversion/FuncToSDFG/FuncToSDFG.h"
#include "mlir/Conversion/LinalgToSDFG/LinalgToSDFG.h"
#include "mlir/Conversion/TensorToSDFG/TensorToSDFG.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/SDFGPasses.h.inc"

} // namespace mlir
