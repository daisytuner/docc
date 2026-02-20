// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern int __docc_test(void)
func.func @test() -> i32 {
// CHECK: int [[a:.*]];
    // CHECK: {
    // CHECK: int [[in:.*]] = 42;
    // CHECK: int [[out:.*]];
    // CHECK: [[out]] = [[in]];
    // CHECK: [[a]] = [[out]];
    // CHECK: }
    %a = arith.constant 42 : i32
    // CHECK: return [[a]]
    func.return %a : i32
}