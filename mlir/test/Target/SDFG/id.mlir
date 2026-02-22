// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern int __docc_test(int [[a:.*]])
sdfg.sdfg @test(%a : i32) -> i32 {
    // CHECK: return [[a]]
    sdfg.return %a : i32
}