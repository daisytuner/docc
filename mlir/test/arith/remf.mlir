// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern float __docc_test(float [[a:.*]], float [[b:.*]])
func.func @test(%a : f32, %b : f32) -> f32 {
// CHECK: float [[c:.*]];
    // CHECK: {
    // CHECK: float [[in1:.*]] = [[a]];
    // CHECK: float [[in2:.*]] = [[b]];
    // CHECK: float [[out:.*]];
    // CHECK: [[out]] = fmod([[in1]], [[in2]]);
    // CHECK: [[c]] = [[out]];
    // CHECK: }
    %c = arith.remf %a, %b : f32
    // CHECK: return [[c]]
    func.return %c : f32
}