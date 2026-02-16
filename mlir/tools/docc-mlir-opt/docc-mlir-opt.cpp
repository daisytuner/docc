#include "mlir/Conversion/SDFGPasses.h"
#include "mlir/Dialect/SDFG/IR/SDFG.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::sdfg::SDFGDialect>();
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();
    mlir::registerSDFGConversionPasses();
    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "DOCC MLIR optimization driver", registry));
}
