#include "mlir/Dialect/Linalg/CustomPasses.h"
#include "mlir/Dialect/Linalg/IR/LinalgCustomOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mlir::linalg::custom::LinalgCustomDialect>();
    mlir::registerAllPasses();
    mlir::registerLinalgCustomPasses();
    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "DOCC MLIR optimization driver", registry));
}
