#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/SDFG/TranslateToSDFG.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char** argv) {
    mlir::registerAllTranslations();
    mlir::sdfg::registerToSDFGTranslation();

    return mlir::failed(mlir::mlirTranslateMain(argc, argv, "DOCC MLIR translation driver"));
}
