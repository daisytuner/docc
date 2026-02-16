// RUN: not docc-mlir-opt %s 2>&1 | FileCheck %s

// CHECK: error: expected '->' in function type

sdfg.sdfg @test(%0 : i32) -> (i32, i32) {
    sdfg.return %0, %0 : (i32, i32)
}