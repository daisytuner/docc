// RUN: not docc-mlir-opt %s 2>&1 | FileCheck %s

// CHECK: error: custom op 'sdfg.sdfg' expected non-empty SDFG body

sdfg.sdfg @test() -> i32 {
}