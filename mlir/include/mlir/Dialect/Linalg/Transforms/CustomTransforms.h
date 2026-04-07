#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace linalg {

void populateLinalgCustomSpecializeGenericOpsPass(RewritePatternSet& patterns);

} // namespace linalg
} // namespace mlir
