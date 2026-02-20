// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern int __docc_test(int [[a:.*]], int [[b:.*]], int [[c:.*]])
sdfg.sdfg @test(%a : i32, %b : i32, %c : i32) -> i32 {
// CHECK: int [[d:.*]];
// CHECK: int [[tmp:.*]];
    %d = sdfg.block -> i32 {
    // CHECK: {
        // CHECK: int [[in1:.*]] = [[a]];
        %in1 = sdfg.memlet %a : i32 -> i32
        // CHECK: int [[in2:.*]] = [[b]];
        %in2 = sdfg.memlet %b : i32 -> i32
        // CHECK: int [[out1:.*]];
        // CHECK: [[out1]] = [[in1]] * [[in2]];
        %out1 = sdfg.tasklet int_mul, %in1, %in2 : (i32, i32) -> i32
        // CHECK: [[tmp]] = [[out1]];
        %tmp = sdfg.memlet %out1 : i32 -> i32
    // CHECK: }
    // CHECK: {
        // CHECK: int [[in3:.*]] = [[tmp]];
        %in3 = sdfg.memlet %tmp : i32 -> i32
        // CHECK: int [[in4:.*]] = [[c]];
        %in4 = sdfg.memlet %c : i32 -> i32
        // CHECK: int [[out2:.*]];
        // CHECK [[out2]] = [[in3]] + [[in4]];
        %out2 = sdfg.tasklet int_add, %in3, %in4 : (i32, i32) -> i32
        // CHECK: [[d]] = [[out2]];
        %d = sdfg.memlet %out2 : i32 -> i32
    // CHECK: }
    }
    // CHECK: return [[d]];
    sdfg.return %d : i32
}