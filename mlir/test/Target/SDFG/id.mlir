// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern int __docc_test(int [[a:.*]])
func.func @test(%a : i32) -> i32 {
    // CHECK: return [[a]]
    func.return %a : i32
}