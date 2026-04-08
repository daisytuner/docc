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

def LinalgCustom_BatchNorm2DNchwOp : LinalgCustom_Op<"batchnorm_2d_nchw",
        [PredOpTrait<"element types must be the same", TCopVTEtAreSameAt<[0, 1, 2, 3, 4, 5, 6]>>]> {
    let summary = "Performs 2-D Batch Normalization";

    let arguments = (ins TensorRankOf<[AnyType], [4]>:$batch,
                         TensorRankOf<[AnyType], [1]>:$e,
                         TensorRankOf<[AnyType], [1]>:$var,
                         TensorRankOf<[AnyType], [1]>:$gamma,
                         TensorRankOf<[AnyType], [1]>:$beta,
                         TensorRankOf<[AnyType], [4]>:$output,
                         TensorRankOf<[AnyType], [0]>:$eps);
    let results = (outs TensorRankOf<[AnyType], [4]>:$result);
    let assemblyFormat = [{
        attr-dict
        `ins` `(`
            $batch `,` $e `,` $var `,` $gamma `,` $beta `,` $eps
            `:`
            type($batch) `,` type($e) `,` type($var) `,` type($gamma) `,` type($beta) `,` type($eps)
        `)`
        `outs` `(` $output `:` type($output) `)`
        `->` type($result)
    }];
    let hasVerifier = 1;
}

#endif // LINALG_CUSTOM_OPS