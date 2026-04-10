#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace linalg {

void populateLinalgCustomSpecializeGenericOpsPass(RewritePatternSet& patterns);

void populateLinalgCustomRemoveRedundantOpsPass(RewritePatternSet& patterns);

void populateLinalgCustomFusePaddingPass(RewritePatternSet& patterns);

} // namespace linalg
} // namespace mlir
