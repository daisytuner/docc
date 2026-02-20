// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern short __docc_test(int [[a:.*]])
func.func @test(%a : i32) -> i16 {
// CHECK: short [[b:.*]];
    // CHECK: {
    // CHECK: int [[in:.*]] = [[a]];
    // CHECK: short [[out:.*]];
    // CHECK: [[out]] = [[in]];
    // CHECK: [[b]] = [[out]];
    // CHECK: }
    %b = arith.trunci %a : i32 to i16
    // CHECK: return [[b]]
    func.return %b : i16
}