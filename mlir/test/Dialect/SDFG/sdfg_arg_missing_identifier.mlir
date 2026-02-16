// RUN: not docc-mlir-opt %s 2>&1 | FileCheck %s

// CHECK: error: custom op 'sdfg.sdfg' expected SSA identifier

sdfg.sdfg @test(%0 : i32, i32)