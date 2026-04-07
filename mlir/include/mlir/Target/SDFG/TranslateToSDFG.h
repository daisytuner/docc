#pragma once

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sdfg {

/// Translate an MLIR operation to a Structured SDFG
LogicalResult translateToSDFG(Operation* op, llvm::raw_ostream& os);

// Register dialects for MLIR to SDFG translation
void dialectRegistration(mlir::DialectRegistry& registry);

/// Register the SDFG translation with MLIR
void registerToSDFGTranslation();

} // namespace sdfg
} // namespace mlir
