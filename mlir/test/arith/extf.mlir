// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern void __docc_test(float [[a:.*]], double *_docc_ret_0)
func.func @test(%a : f32) -> f64 {
// CHECK: double [[b:.*]];
    // CHECK: {
    // CHECK: float [[in:.*]] = [[a]];
    // CHECK: double [[out:.*]];
    // CHECK: [[out]] = [[in]];
    // CHECK: [[b]] = [[out]];
    // CHECK: }
    %b = arith.extf %a : f32 to f64
    // CHECK: {{.*}}_docc_ret_0)[0] = {{.*}}
    // CHECK: return ;
    func.return %b : f64
}
