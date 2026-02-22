#ifndef SDFG_CONVERSION_PASSES
#define SDFG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertArithToSDFG : Pass<"convert-arith-to-sdfg"> {
    let summary = "Convert Arith dialect to SDFG dialect";
    let dependentDialects = ["::mlir::sdfg::SDFGDialect"];
}

def ConvertToSDFG : Pass<"convert-to-sdfg"> {
    let summary = "Convert to SDFG";
    let description = [{This is a generic pass to convert to SDFG dialect.}];
    let dependentDialects = ["::mlir::sdfg::SDFGDialect"];
}

def ConvertFuncToSDFG : Pass<"convert-func-to-sdfg"> {
    let summary = "Convert Func dialect to SDFG dialect";
    let dependentDialects = ["::mlir::sdfg::SDFGDialect"];
}

def ConvertLinalgToSDFG : Pass<"convert-linalg-to-sdfg"> {
    let summary = "Convert Linalg dialect to SDFG dialect";
    let dependentDialects = ["::mlir::sdfg::SDFGDialect"];
}

def ConvertTensorToSDFG : Pass<"convert-tensor-to-sdfg"> {
    let summary = "Convert Tensor dialect to SDFG dialect";
    let dependentDialects = ["::mlir::sdfg::SDFGDialect"];
}

#endif // SDFG_CONVERSION_PASSES