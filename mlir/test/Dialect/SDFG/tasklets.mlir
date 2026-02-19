// RUN: docc-mlir-opt %s | docc-mlir-opt | FileCheck %s

// CHECK-LABEL: @test_fp_assign
sdfg.sdfg @test_fp_assign(%0 : f32) -> f32 {
    %1 = sdfg.block -> f32 {
        %2 = sdfg.memlet %0 : f32 -> f32
        %3 = sdfg.tasklet assign, %2 : (f32) -> f32
        %4 = sdfg.memlet %3 : f32 -> f32
        sdfg.yield %4 : f32
    }
    sdfg.return %1 : f32
}

// CHECK-LABEL: @test_int_assign
sdfg.sdfg @test_int_assign(%0 : i32) -> i32 {
    %1 = sdfg.block -> i32 {
        %2 = sdfg.memlet %0 : i32 -> i32
        %3 = sdfg.tasklet assign, %2 : (i32) -> i32
        %4 = sdfg.memlet %3 : i32 -> i32
        sdfg.yield %4 : i32
    }
    sdfg.return %1 : i32
}

// CHECK-LABEL: @test_fp_neg
sdfg.sdfg @test_fp_neg(%0 : f32) -> f32 {
    %1 = sdfg.block -> f32 {
        %2 = sdfg.memlet %0 : f32 -> f32
        %3 = sdfg.tasklet fp_neg, %2 : (f32) -> f32
        %4 = sdfg.memlet %3 : f32 -> f32
        sdfg.yield %4 : f32
    }
    sdfg.return %1 : f32
}

// CHECK-LABEL: @test_fp_add
sdfg.sdfg @test_fp_add(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_add, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_sub
sdfg.sdfg @test_fp_sub(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_sub, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_mul
sdfg.sdfg @test_fp_mul(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_mul, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_div
sdfg.sdfg @test_fp_div(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_div, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_rem
sdfg.sdfg @test_fp_rem(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_rem, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_fma
sdfg.sdfg @test_fp_fma(%0 : f32, %1 : f32, %2 : f32) -> f32 {
    %3 = sdfg.block -> f32 {
        %4 = sdfg.memlet %0 : f32 -> f32
        %5 = sdfg.memlet %1 : f32 -> f32
        %6 = sdfg.memlet %2 : f32 -> f32
        %7 = sdfg.tasklet fp_fma, %4, %5, %6 : (f32, f32, f32) -> f32
        %8 = sdfg.memlet %7 : f32 -> f32
        sdfg.yield %8 : f32
    }
    sdfg.return %3 : f32
}

// CHECK-LABEL: @test_fp_oeq
sdfg.sdfg @test_fp_oeq(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_oeq, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_one
sdfg.sdfg @test_fp_one(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_one, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_oge
sdfg.sdfg @test_fp_oge(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_oge, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_ogt
sdfg.sdfg @test_fp_ogt(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_ogt, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_ole
sdfg.sdfg @test_fp_ole(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_ole, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_olt
sdfg.sdfg @test_fp_olt(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_olt, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_ord
sdfg.sdfg @test_fp_ord(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_ord, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_ueq
sdfg.sdfg @test_fp_ueq(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_ueq, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_une
sdfg.sdfg @test_fp_une(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_une, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_ugt
sdfg.sdfg @test_fp_ugt(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_ugt, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_uge
sdfg.sdfg @test_fp_uge(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_uge, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_ult
sdfg.sdfg @test_fp_ult(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_ult, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_ule
sdfg.sdfg @test_fp_ule(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_ule, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
}

// CHECK-LABEL: @test_fp_uno
sdfg.sdfg @test_fp_uno(%0 : f32, %1 : f32) -> f32 {
    %2 = sdfg.block -> f32 {
        %3 = sdfg.memlet %0 : f32 -> f32
        %4 = sdfg.memlet %1 : f32 -> f32
        %5 = sdfg.tasklet fp_uno, %3, %4 : (f32, f32) -> f32
        %6 = sdfg.memlet %5 : f32 -> f32
        sdfg.yield %6 : f32
    }
    sdfg.return %2 : f32
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

// CHECK-LABEL: @test_int_sub
sdfg.sdfg @test_int_sub(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_sub, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_mul
sdfg.sdfg @test_int_mul(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_mul, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_sdiv
sdfg.sdfg @test_int_sdiv(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_sdiv, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_srem
sdfg.sdfg @test_int_srem(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_srem, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_udiv
sdfg.sdfg @test_int_udiv(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_udiv, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_urem
sdfg.sdfg @test_int_urem(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_urem, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_and
sdfg.sdfg @test_int_and(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_and, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_or
sdfg.sdfg @test_int_or(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_or, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_xor
sdfg.sdfg @test_int_xor(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_xor, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_shl
sdfg.sdfg @test_int_shl(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_shl, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_ashr
sdfg.sdfg @test_int_ashr(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_ashr, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_lshr
sdfg.sdfg @test_int_lshr(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_lshr, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_smin
sdfg.sdfg @test_int_smin(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_smin, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_smax
sdfg.sdfg @test_int_smax(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_smax, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_scmp
sdfg.sdfg @test_int_scmp(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_scmp, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_umin
sdfg.sdfg @test_int_umin(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_umin, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_umax
sdfg.sdfg @test_int_umax(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_umax, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_ucmp
sdfg.sdfg @test_int_ucmp(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_ucmp, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_eq
sdfg.sdfg @test_int_eq(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_eq, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_ne
sdfg.sdfg @test_int_ne(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_ne, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_sge
sdfg.sdfg @test_int_sge(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_sge, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_sgt
sdfg.sdfg @test_int_sgt(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_sgt, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_sle
sdfg.sdfg @test_int_sle(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_sle, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_slt
sdfg.sdfg @test_int_slt(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_slt, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_uge
sdfg.sdfg @test_int_uge(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_uge, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_ugt
sdfg.sdfg @test_int_ugt(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_ugt, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_ule
sdfg.sdfg @test_int_ule(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_ule, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_ult
sdfg.sdfg @test_int_ult(%0 : i32, %1 : i32) -> i32 {
    %2 = sdfg.block -> i32 {
        %3 = sdfg.memlet %0 : i32 -> i32
        %4 = sdfg.memlet %1 : i32 -> i32
        %5 = sdfg.tasklet int_ult, %3, %4 : (i32, i32) -> i32
        %6 = sdfg.memlet %5 : i32 -> i32
        sdfg.yield %6 : i32
    }
    sdfg.return %2 : i32
}

// CHECK-LABEL: @test_int_abs
sdfg.sdfg @test_int_abs(%0 : i32) -> i32 {
    %1 = sdfg.block -> i32 {
        %2 = sdfg.memlet %0 : i32 -> i32
        %3 = sdfg.tasklet int_abs, %2 : (i32) -> i32
        %4 = sdfg.memlet %3 : i32 -> i32
        sdfg.yield %4 : i32
    }
    sdfg.return %1 : i32
}