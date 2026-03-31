// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern void __docc_test(int [[a:.*]], short *_docc_ret_0)
func.func @test(%a : i32) -> i16 {
// CHECK: short [[b:.*]];
    // CHECK: {
    // CHECK: int [[in:.*]] = [[a]];
    // CHECK: short [[out:.*]];
    // CHECK: [[out]] = [[in]];
    // CHECK: [[b]] = [[out]];
    // CHECK: }
    %b = arith.trunci %a : i32 to i16
    // CHECK: {{.*}}_docc_ret_0)[0] = {{.*}}
    // CHECK: return ;
    func.return %b : i16
}
