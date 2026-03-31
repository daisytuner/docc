// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern void __docc_test(int [[a:.*]], int [[b:.*]], bool *_docc_ret_0)
func.func @test(%a : i32, %b : i32) -> i1 {
// CHECK: bool [[c:.*]];
    // CHECK: {
    // CHECK: int [[in1:.*]] = [[a]];
    // CHECK: int [[in2:.*]] = [[b]];
    // CHECK: bool [[out:.*]];
    // CHECK: [[out]] = [[in1]] < [[in2]];
    // CHECK: [[c]] = [[out]];
    // CHECK: }
    %c = arith.cmpi slt, %a, %b : i32
    // CHECK: {{.*}}_docc_ret_0)[0] = {{.*}}
    // CHECK: return ;
    func.return %c : i1
}
