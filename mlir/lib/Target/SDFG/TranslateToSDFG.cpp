#include "mlir/Target/SDFG/TranslateToSDFG.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Target/SDFG/SDFGTranslator.h"
#include "mlir/Tools/mlir-translate/Translation.h"

namespace mlir {
namespace sdfg {

void registerToSDFGTranslation() {
    TranslateFromMLIRRegistration registration(
        "mlir-to-sdfg",
        "translate MLIR to Structured SDFG",
        [](Operation* op, raw_ostream& os) {
            SDFGTranslator translator;
            if (failed(translateOp(translator, op))) {
                emitError(op->getLoc(), "Could not translate to SDFG");
                return failure();
            }

            if (failed(emitJSON(translator, os))) {
                emitError(op->getLoc(), "Could not generate code");
                return failure();
            }

            return success();
        },
        [](mlir::DialectRegistry& registry) {
            registry.insert<func::FuncDialect>();
            registry.insert<arith::ArithDialect>();
            registry.insert<linalg::LinalgDialect>();
            registry.insert<tensor::TensorDialect>();
        }
    );
}

} // namespace sdfg
} // namespace mlir
