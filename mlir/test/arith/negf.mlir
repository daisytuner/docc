// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern void __docc_test(float [[a:.*]], float *_docc_ret_0)
func.func @test(%a : f32) -> f32 {
// CHECK: float [[b:.*]];
    // CHECK: {
    // CHECK: float [[in:.*]] = [[a]];
    // CHECK: float [[out:.*]];
    // CHECK: [[out]] = -[[in]];
    // CHECK: [[b]] = [[out]];
    // CHECK: }
    %b = arith.negf %a : f32
    // CHECK: {{.*}}_docc_ret_0)[0] = {{.*}}
    // CHECK: return ;
    func.return %b : f32
}
