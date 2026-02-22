#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/SDFG/TranslateToSDFG.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include <sdfg/serializer/json_serializer.h>

int main(int argc, char** argv) {
    // Register SDFG library node serializers
    ::sdfg::serializer::register_default_serializers();

    mlir::registerAllTranslations();
    mlir::sdfg::registerToSDFGTranslation();

    return mlir::failed(mlir::mlirTranslateMain(argc, argv, "DOCC MLIR translation driver"));
}
