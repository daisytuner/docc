// RUN: docc-mlir-opt %s | docc-mlir-opt | FileCheck %s

// CHECK-LABEL: @test_sdfg
sdfg.sdfg @test_sdfg()

// CHECK-LABEL: @test_ext_sdfg_with_args
sdfg.sdfg @test_ext_sdfg_with_args(i32, f32, i64) -> i32

// CHECK-LABEL: @test_ext_sdfg_with_args_and_names
sdfg.sdfg @test_ext_sdfg_with_args_and_names(%0 : i32, %1 : f32, %2 : i64) -> i32

// CHECK-LABEL: @test_empty_return
sdfg.sdfg @test_empty_return() {
    sdfg.return
}

// CHECK-LABEL: @test_identity
sdfg.sdfg @test_identity(%0: i32) -> i32 {
    sdfg.return %0 : i32
}

// CHECK-LABEL: @test_empty_block
sdfg.sdfg @test_empty_block() {
    sdfg.block
    sdfg.return
}

// CHECK-LABEL: @test_block_yield
sdfg.sdfg @test_block_yield() -> i32 {
    %0 = sdfg.block -> i32 {
        %1 = sdfg.constant 1 : i32
        sdfg.yield %1 : i32
    }
    sdfg.return %0 : i32
}

// CHECK-LABEL: @test_block_yield_multi_dim
sdfg.sdfg @test_block_yield_multi_dim() -> i32 {
    %0, %1 = sdfg.block -> (i32, f32) {
        %2 = sdfg.constant 1 : i32
        %3 = sdfg.constant 2.0 : f32
        sdfg.yield %2, %3 : i32, f32
    }
    sdfg.return %0 : i32
}

// CHECK-LABEL: @test_block_implicit_yield
sdfg.sdfg @test_block_implicit_yield() -> i32 {
    %0 = sdfg.block -> i32 {
        %1 = sdfg.constant 1 : i32
        // CHECK: sdfg.yield
    }
    sdfg.return %0 : i32
}

// CHECK-LABEL: @test_block_implicit_yield_multi_dim
sdfg.sdfg @test_block_implicit_yield_multi_dim() -> i32 {
    %0, %1 = sdfg.block -> (i32, f32) {
        %2 = sdfg.constant 1 : i32
        %3 = sdfg.constant 2.0 : f32
        // CHECK: sdfg.yield
    }
    sdfg.return %0 : i32
}

// CHECK-LABEL: @test_constant
sdfg.sdfg @test_constant() -> i32 {
    %0 = sdfg.block -> i32 {
        %1 = sdfg.constant 1 : i32
        sdfg.yield %1 : i32
    }
    sdfg.return %0 : i32
}

// CHECK-LABEL: @test_tasklet_assign
sdfg.sdfg @test_tasklet_assign() -> i32 {
    %0 = sdfg.block -> i32 {
        %1 = sdfg.constant 1 : i32
        %2 = sdfg.tasklet assign, %1 : (i32) -> i32
        %3 = sdfg.memlet %2 : i32 -> i32
        sdfg.yield %3 : i32
    }
    sdfg.return %0 : i32
}

// CHECK-LABEL: @test_tasklet_int_add
sdfg.sdfg @test_tasklet_int_add() -> i32 {
    %0 = sdfg.block -> i32 {
        %1 = sdfg.constant 1 : i32
        %2 = sdfg.constant 2 : i32
        %3 = sdfg.tasklet int_add, %1, %2 : (i32, i32) -> i32
        %4 = sdfg.memlet %3 : i32 -> i32
        sdfg.yield %4 : i32
    }
    sdfg.return %0 : i32
}

// CHECK-LABEL: @test_tasklet_int_add2
sdfg.sdfg @test_tasklet_int_add2() -> i32 {
    %0 = sdfg.block -> i32 {
        %1 = sdfg.constant 1 : i32
        %2 = sdfg.constant 2 : i16
        %3 = sdfg.tasklet int_add, %1, %2 : (i32, i16) -> i32
        %4 = sdfg.memlet %3 : i32 -> i32
        sdfg.yield %4 : i32
    }
    sdfg.return %0 : i32
}

// CHECK-LABEL: @test_tasklet_fp_fma
sdfg.sdfg @test_tasklet_fp_fma() -> f32 {
    %0 = sdfg.block -> f32 {
        %1 = sdfg.constant 1.0 : f32
        %2 = sdfg.constant 2.0 : f32
        %3 = sdfg.constant 3.0 : f32
        %4 = sdfg.tasklet fp_fma, %1, %2, %3 : (f32, f32, f32) -> f32
        %5 = sdfg.memlet %4 : f32 -> f32
        sdfg.yield %5 : f32
    }
    sdfg.return %0 : f32
}

// CHECK-LABEL: @test_int_add
sdfg.sdfg @test_int_add(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_add, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_malloc
module {
  sdfg.sdfg @test_malloc () -> tensor<4x4xf32> {
    %0 = sdfg.block -> tensor<4x4xf32> {
      %1 = sdfg.malloc : tensor<4x4xf32>
      sdfg.yield %1 : tensor<4x4xf32>
    }
    sdfg.return %0 : tensor<4x4xf32>
  }
}
