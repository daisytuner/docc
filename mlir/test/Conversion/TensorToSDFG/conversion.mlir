// RUN: docc-mlir-opt %s --convert-to-sdfg | FileCheck %s

// CHECK: module {
// CHECK:   sdfg.sdfg @test_malloc () -> tensor<4x4xf32> {
// CHECK:     %[[VAL0:.*]] = sdfg.block -> tensor<4x4xf32> {
// CHECK:       %[[VAL1:.*]] = sdfg.malloc : tensor<4x4xf32>
// CHECK:       sdfg.yield %[[VAL1]] : tensor<4x4xf32>
// CHECK:     }
// CHECK:     sdfg.return %[[VAL0]] : tensor<4x4xf32>
// CHECK:   }
// CHECK: }

func.func @test_malloc() -> tensor<4x4xf32> {
   %0 = tensor.empty() : tensor<4x4xf32>
   func.return %0 : tensor<4x4xf32>
}