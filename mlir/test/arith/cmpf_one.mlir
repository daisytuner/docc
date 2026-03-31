// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern void __docc_test(float [[a:.*]], float [[b:.*]], bool *_docc_ret_0)
func.func @test(%a : f32, %b : f32) -> i1 {
// CHECK: bool [[c:.*]];
    // CHECK: {
    // CHECK: float [[in1:.*]] = [[a]];
    // CHECK: float [[in2:.*]] = [[b]];
    // CHECK: bool [[out:.*]];
    // CHECK: [[out]] = [[in1]] != [[in2]];
    // CHECK: [[c]] = [[out]];
    // CHECK: }
    %c = arith.cmpf one, %a, %b : f32
    // CHECK: {{.*}}_docc_ret_0)[0] = {{.*}}
    // CHECK: return ;
    func.return %c : i1
}
