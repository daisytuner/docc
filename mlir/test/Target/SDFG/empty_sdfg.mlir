// RUN: docc-mlir-translate --mlir-to-sdfg %s | sdfg-json-to-c > %t
// RUN: FileCheck %s < %t

// CHECK: extern void __docc_test(void)
sdfg.sdfg @test() {
    // CHECK-NOT: return
    sdfg.return
}