#ifndef MLIR_DIALECT_LINALG_CUSTOM_PASSES
#define MLIR_DIALECT_LINALG_CUSTOM_PASSES

include "mlir/Pass/PassBase.td"

def LinalgCustomSpecializeGenericOpsPass : Pass<"linalg-custom-specialize-generic-ops"> {
    let summary = "Convert generic ops back to named ops (custom DOCC MLIR pass)";
    let description = [{
        Custom DOCC MLIR pass that support more than the pass from core MLIR.
    }];
    let dependentDialects = [
        "linalg::LinalgDialect",
        "linalg::custom::LinalgCustomDialect"
    ];
}

#endif // MLIR_DIALECT_LINALG_CUSTOM_PASSES