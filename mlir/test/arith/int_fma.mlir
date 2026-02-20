// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern int __docc_test(int [[a:.*]], int [[b:.*]], int [[c:.*]])
func.func @test(%a : i32, %b : i32, %c : i32) -> i32 {
// CHECK: int [[d:.*]];
// CHECK: int [[tmp:.*]];
    // CHECK: {
    // CHECK: int [[in1:.*]] = [[a]];
    // CHECK: int [[in2:.*]] = [[b]];
    // CHECK: int [[out1:.*]];
    // CHECK: [[out1]] = [[in1]] * [[in2]];
    // CHECK: [[tmp]] = [[out1]];
    // CHECK: }
    %tmp = arith.muli %a, %b : i32
    // CHECK: {
    // CHECK: int [[in3:.*]] = [[tmp]];
    // CHECK: int [[in4:.*]] = [[c]];
    // CHECK: int [[out2:.*]];
    // CHECK [[out2]] = [[in3]] + [[in4]];
    // CHECK: [[d]] = [[out2]];
    // CHECK: }
    %d = arith.addi %tmp, %c : i32
    // CHECK: return [[d]];
    func.return %d : i32
}