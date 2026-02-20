// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern int __docc_test(float [[a:.*]])
func.func @test(%a : f32) -> i32 {
// CHECK: int [[b:.*]];
    // CHECK: {
    // CHECK: float [[in:.*]] = [[a]];
    // CHECK: int [[out:.*]];
    // CHECK: [[out]] = [[in]];
    // CHECK: [[b]] = [[out]];
    // CHECK: }
    %b = arith.fptosi %a : f32 to i32
    // CHECK: return [[b]]
    func.return %b : i32
}