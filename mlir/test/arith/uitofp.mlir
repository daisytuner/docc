// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern float __docc_test(int [[a:.*]])
func.func @test(%a : i32) -> f32 {
// CHECK: float [[b:.*]];
    // CHECK: {
    // CHECK: int [[in:.*]] = [[a]];
    // CHECK: float [[out:.*]];
    // CHECK: [[out]] = [[in]];
    // CHECK: [[b]] = [[out]];
    // CHECK: }
    %b = arith.uitofp %a : i32 to f32
    // CHECK: return [[b]]
    func.return %b : f32
}