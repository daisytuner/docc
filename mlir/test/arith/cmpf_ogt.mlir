// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern bool __docc_test(float [[a:.*]], float [[b:.*]])
func.func @test(%a : f32, %b : f32) -> i1 {
// CHECK: bool [[c:.*]];
    // CHECK: {
    // CHECK: float [[in1:.*]] = [[a]];
    // CHECK: float [[in2:.*]] = [[b]];
    // CHECK: bool [[out:.*]];
    // CHECK: [[out]] = [[in1]] > [[in2]];
    // CHECK: [[c]] = [[out]];
    // CHECK: }
    %c = arith.cmpf ogt, %a, %b : f32
    // CHECK: return [[c]]
    func.return %c : i1
}