// RUN: not docc-mlir-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'sdfg.return' op has operand, but enclosing SDFG (@test) has no result type

sdfg.sdfg @test(%0 : i32) {
    sdfg.return %0 : i32
}