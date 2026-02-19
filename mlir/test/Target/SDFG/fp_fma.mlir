// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern float __docc_test(float [[a:.*]], float [[b:.*]], float [[c:.*]])
sdfg.sdfg @test(%a : f32, %b : f32, %c : f32) -> f32 {
// CHECK: float [[d:.*]];
    // CHECK: {
    %d = sdfg.block -> f32 {
        // CHECK: float [[in1:.*]] = [[a]];
        %in1 = sdfg.memlet %a : f32 -> f32
        // CHECK: float [[in2:.*]] = [[b]];
        %in2 = sdfg.memlet %b : f32 -> f32
        // CHECK: float [[in3:.*]] = [[c]];
        %in3 = sdfg.memlet %c : f32 -> f32
        // CHECK: float [[out:.*]];
        // CHECK: [[out]] = [[in1]] * [[in2]] + [[in3]];
        %out = sdfg.tasklet fp_fma, %in1, %in2, %in3 : (f32, f32, f32) -> f32
        // CHECK: [[d]] = [[out]];
        %d = sdfg.memlet %out : f32 -> f32
        sdfg.yield %d : f32
    // CHECK: }
    }
    // CHECK: return [[d]];
    sdfg.return %d : f32
}