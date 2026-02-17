// RUN: not docc-mlir-opt %s 2>&1 | FileCheck %s

// CHECK: error: custom op 'sdfg.sdfg' expected type instead of SSA identifier

sdfg.sdfg @test(i32, %0 : i32)