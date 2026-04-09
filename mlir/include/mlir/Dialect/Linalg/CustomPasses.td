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

def LinalgCustomRemoveRedundantOpsPass : Pass<"linalg-custom-remove-redundant-ops"> {
    let summary = "Remove redundant ops (custom DOCC MLIR pass)";
    let dependentDialects = [
        "linalg::LinalgDialect"
    ];
}

def LinalgCustomFuseConvPaddingPass : Pass<"linalg-custom-fuse-conv-padding"> {
    let summary = "Fuse tensor.pad operations into convolutions (custom DOCC MLIR pass)";
    let dependentDialects = [
        "arith::ArithDialect",
        "linalg::LinalgDialect",
        "linalg::custom::LinalgCustomDialect",
        "tensor::TensorDialect"
    ];
}

#endif // MLIR_DIALECT_LINALG_CUSTOM_PASSES