// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern int __docc_test(int [[a:.*]], int [[b:.*]])
sdfg.sdfg @test(%a : i32, %b : i32) -> i32 {
// CHECK: int [[c:.*]];
    // CHECK: {
    %c = sdfg.block -> i32 {
        // CHECK: int [[in1:.*]] = [[a]];
        %in1 = sdfg.memlet %a : i32 -> i32
        // CHECK: int [[in2:.*]] = [[b]];
        %in2 = sdfg.memlet %b : i32 -> i32
        // CHECK: int [[out:.*]];
        // CHECK: [[out]] = [[in1]] + [[in2]];
        %out = sdfg.tasklet int_add, %in1, %in2 : (i32, i32) -> i32
        // CHECK: [[c]] = [[out]];
        %c = sdfg.memlet %out : i32 -> i32
    // CHECK: }
    }
    // CHECK: return [[c]]
    sdfg.return %c : i32
}