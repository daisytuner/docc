#ifndef LINALG_CUSTOM_OPS
#define LINALG_CUSTOM_OPS

include "mlir/IR/OpBase.td"

def LinalgCustom_Dialect : Dialect {
    let name = "linalg_custom";
    let description = [{
        Extension for the linalg dialect with custom operations.
    }];
    let cppNamespace = "::mlir::linalg::custom";
    let dependentDialects = [
        "linalg::LinalgDialect"
    ];
}

class LinalgCustom_Op<string mnemonic, list<Trait> traits = []> : Op<LinalgCustom_Dialect, mnemonic, traits>;

def LinalgCustom_ReLUOp : LinalgCustom_Op<"relu",
        [PredOpTrait<"input and output have same element type", TCopVTEtIsSameAs<0, 1>>]> {
    let summary = "ReLU operator";

    let arguments = (ins AnyShaped:$input,
                         AnyShaped:$output);
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        attr-dict
        `ins` `(` $input `:` type($input) `)`
        `outs` `(` $output `:` type($output) `)`
        `->` type($result)
    }];
    let hasVerifier = 1;
}

def LinalgCustom_SigmoidOp : LinalgCustom_Op<"sigmoid",
        [PredOpTrait<"input and output have same element type", TCopVTEtIsSameAs<0, 1>>]> {
    let summary = "Sigmoid operator";

    let arguments = (ins AnyShaped:$input,
                         AnyShaped:$output);
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = [{
        attr-dict
        `ins` `(` $input `:` type($input) `)`
        `outs` `(` $output `:` type($output) `)`
        `->` type($result)
    }];
    let hasVerifier = 1;
}

#endif // LINALG_CUSTOM_OPS