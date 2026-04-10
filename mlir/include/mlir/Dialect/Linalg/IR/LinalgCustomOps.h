#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"

//===----------------------------------------------------------------------===//
// Linalg Custom Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgCustomOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Linalg Custom Dialect Enum Attributes
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgCustomOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// Linalg Custom Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgCustomOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgCustomOps.h.inc"
