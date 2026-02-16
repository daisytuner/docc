// RUN: not docc-mlir-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'sdfg.return' op has no operand, but enclosing SDFG (@test) has a result type

sdfg.sdfg @test() -> i32 {
    sdfg.return
}