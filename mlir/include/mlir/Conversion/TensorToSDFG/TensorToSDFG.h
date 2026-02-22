#pragma once

#include <memory>

namespace mlir {

class DialectRegistry;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTTENSORTOSDFG
#include "mlir/Conversion/SDFGPasses.h.inc"

namespace tensor {

void populateTensorToSDFGPatterns(RewritePatternSet& patterns);

} // namespace tensor
} // namespace mlir
