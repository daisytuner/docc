// RUN: not docc-mlir-opt %s 2>&1 | FileCheck %s

// CHECK: error: type of return operand ('i32') doesn't match SDFG result type ('i16') in SDFG @test

sdfg.sdfg @test(%0 : i32) -> i16 {
    sdfg.return %0 : i32
}