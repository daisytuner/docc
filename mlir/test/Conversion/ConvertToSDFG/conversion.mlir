// RUN: docc-mlir-opt %s --convert-to-sdfg > %t
// RUN: FileCheck %s < %t

// CHECK: sdfg.sdfg @test_i32_add (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @test_i32_add(%0 : i32, %1 : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_add, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %2 = arith.addi %0, %1 : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    func.return %2 : i32
}