// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern void __docc_test(int [[a:.*]], int [[b:.*]], int *_docc_ret_0)
func.func @test(%a : i32, %b : i32) -> i32 {
// CHECK: int [[c:.*]];
    // CHECK: {
    // CHECK: int [[in1:.*]] = [[a]];
    // CHECK: int [[in2:.*]] = [[b]];
    // CHECK: int [[out:.*]];
    // CHECK: [[out]] = [[in1]] / [[in2]];
    // CHECK: [[c]] = [[out]];
    // CHECK: }
    %c = arith.divsi %a, %b : i32
    // CHECK: {{.*}}_docc_ret_0)[0] = {{.*}}
    // CHECK: return ;
    func.return %c : i32
}
