// RUN: docc-mlir-opt %s --convert-func-to-sdfg > %t
// RUN: FileCheck %s < %t

// CHECK: sdfg.sdfg @test_empty ()
func.func @test_empty() {
    // CHECK: sdfg.return
    func.return
}

// CHECK: sdfg.sdfg @test_i32_add (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @test_i32_add(%0 : i32, %1 : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
    %2 = arith.addi %0, %1 : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    func.return %2 : i32
}