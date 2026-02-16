#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::sdfg::SDFGDialect>();
    mlir::registerAllDialects(registry);
    return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}