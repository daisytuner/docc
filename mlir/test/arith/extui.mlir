// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern long long __docc_test(int [[a:.*]])
func.func @test(%a : i32) -> i64 {
// CHECK: long long [[b:.*]];
    // CHECK: {
    // CHECK: int [[in:.*]] = [[a]];
    // CHECK: long long [[out:.*]];
    // CHECK: [[out]] = [[in]];
    // CHECK: [[b]] = [[out]];
    // CHECK: }
    %b = arith.extui %a : i32 to i64
    // CHECK: return [[b]]
    func.return %b : i64
}