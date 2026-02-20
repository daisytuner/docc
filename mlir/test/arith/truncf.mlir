// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern float __docc_test(double [[a:.*]])
func.func @test(%a : f64) -> f32 {
// CHECK: float [[b:.*]];
    // CHECK: {
    // CHECK: double [[in:.*]] = [[a]];
    // CHECK: float [[out:.*]];
    // CHECK: [[out]] = [[in]];
    // CHECK: [[b]] = [[out]];
    // CHECK: }
    %b = arith.truncf %a : f64 to f32
    // CHECK: return [[b]]
    func.return %b : f32
}