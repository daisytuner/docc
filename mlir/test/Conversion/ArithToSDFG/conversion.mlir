// RUN: docc-mlir-opt %s --convert-arith-to-sdfg > %t
// RUN: FileCheck %s < %t

// CHECK: sdfg.sdfg @test_addf (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> f32
sdfg.sdfg @test_addf(%a : f32, %b : f32) -> f32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_add, %[[VAR1]], %[[VAR2]] : (f32, f32) -> f32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : f32 -> f32
    // CHECK: sdfg.yield %[[VAR4]] : f32
    %c = arith.addf %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : f32
    sdfg.return %c : f32
}

// CHECK: sdfg.sdfg @test_addi (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_addi(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_add, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.addi %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_andi (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_andi(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_and, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.andi %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_bitcast (%[[ARG0:.*]]: i32) -> f32
sdfg.sdfg @test_bitcast(%a : i32) -> f32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (i32) -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : f32 -> f32
    // CHECK: sdfg.yield %[[VAR3]] : f32
    %b = arith.bitcast %a : i32 to f32
    // CHECK: sdfg.return %[[VAR0]] : f32
    sdfg.return %b : f32
}

// CHECK: sdfg.sdfg @test_cmpf_false (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_false(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.constant false
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (i1) -> i1
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR3]] : i1
    %c = arith.cmpf false, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_oeq (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_oeq(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_oeq, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf oeq, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_ogt (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_ogt(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_ogt, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf ogt, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_oge (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_oge(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_oge, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf oge, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_olt (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_olt(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_olt, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf olt, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_ole (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_ole(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_ole, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf ole, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_one (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_one(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_one, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf one, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_ord (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_ord(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_ord, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf ord, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_ueq (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_ueq(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_ueq, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf ueq, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_ugt (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_ugt(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_ugt, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf ugt, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_uge (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_uge(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_uge, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf uge, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_ult (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_ult(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_ult, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf ult, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_ule (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_ule(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_ule, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf ule, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_une (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_une(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_une, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf une, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_uno (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_uno(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_uno, %[[VAR1]], %[[VAR2]] : (f32, f32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpf uno, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpf_true (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> i1
sdfg.sdfg @test_cmpf_true(%a : f32, %b : f32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.constant true
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (i1) -> i1
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR3]] : i1
    %c = arith.cmpf true, %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_eq (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_eq(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_eq, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi eq, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_ne (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_ne(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_ne, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi ne, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_slt (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_slt(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_slt, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi slt, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_sle (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_sle(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_sle, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi sle, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_sgt (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_sgt(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_sgt, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi sgt, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_sge (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_sge(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_sge, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi sge, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_ult (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_ult(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_ult, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi ult, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_ule (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_ule(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_ule, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi ule, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_ugt (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_ugt(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_ugt, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi ugt, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_cmpi_uge (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i1
sdfg.sdfg @test_cmpi_uge(%a : i32, %b : i32) -> i1 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i1
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_uge, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i1
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i1 -> i1
    // CHECK: sdfg.yield %[[VAR4]] : i1
    %c = arith.cmpi uge, %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i1
    sdfg.return %c : i1
}

// CHECK: sdfg.sdfg @test_extf (%[[ARG0:.*]]: f32) -> f64
sdfg.sdfg @test_extf(%a : f32) -> f64 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f64
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (f32) -> f64
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : f64 -> f64
    // CHECK: sdfg.yield %[[VAR3]] : f64
    %b = arith.extf %a : f32 to f64
    // CHECK: sdfg.return %[[VAR0]] : f64
    sdfg.return %b : f64
}

// CHECK: sdfg.sdfg @test_extsi (%[[ARG0:.*]]: i32) -> i64
sdfg.sdfg @test_extsi(%a : i32) -> i64 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i64
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (i32) -> i64
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : i64 -> i64
    // CHECK: sdfg.yield %[[VAR3]] : i64
    %b = arith.extsi %a : i32 to i64
    // CHECK: sdfg.return %[[VAR0]] : i64
    sdfg.return %b : i64
}

// CHECK: sdfg.sdfg @test_extui (%[[ARG0:.*]]: i32) -> i64
sdfg.sdfg @test_extui(%a : i32) -> i64 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i64
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (i32) -> i64
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : i64 -> i64
    // CHECK: sdfg.yield %[[VAR3]] : i64
    %b = arith.extui %a : i32 to i64
    // CHECK: sdfg.return %[[VAR0]] : i64
    sdfg.return %b : i64
}

// CHECK: sdfg.sdfg @test_fptosi (%[[ARG0:.*]]: f32) -> i32
sdfg.sdfg @test_fptosi(%a : f32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (f32) -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR3]] : i32
    %b = arith.fptosi %a : f32 to i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %b : i32
}

// CHECK: sdfg.sdfg @test_fptoui (%[[ARG0:.*]]: f32) -> i32
sdfg.sdfg @test_fptoui(%a : f32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (f32) -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR3]] : i32
    %b = arith.fptoui %a : f32 to i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %b : i32
}

// CHECK: sdfg.sdfg @test_maxsi (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_maxsi(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_smax, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.maxsi %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_maxui (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_maxui(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_umax, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.maxui %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_minsi (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_minsi(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_smin, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.minsi %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_minui (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_minui(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_umin, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.minui %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_mulf (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> f32
sdfg.sdfg @test_mulf(%a : f32, %b : f32) -> f32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_mul, %[[VAR1]], %[[VAR2]] : (f32, f32) -> f32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : f32 -> f32
    // CHECK: sdfg.yield %[[VAR4]] : f32
    %c = arith.mulf %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : f32
    sdfg.return %c : f32
}

// CHECK: sdfg.sdfg @test_muli (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_muli(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_mul, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.muli %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_negf (%[[ARG0:.*]]: f32) -> f32
sdfg.sdfg @test_negf(%a : f32) -> f32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet fp_neg, %[[VAR1]] : (f32) -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : f32 -> f32
    // CHECK: sdfg.yield %[[VAR3]] : f32
    %b = arith.negf %a : f32
    // CHECK: sdfg.return %[[VAR0]] : f32
    sdfg.return %b : f32
}

// CHECK: sdfg.sdfg @test_ori (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_ori(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_or, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.ori %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_remf (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> f32
sdfg.sdfg @test_remf(%a : f32, %b : f32) -> f32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_rem, %[[VAR1]], %[[VAR2]] : (f32, f32) -> f32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : f32 -> f32
    // CHECK: sdfg.yield %[[VAR4]] : f32
    %c = arith.remf %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : f32
    sdfg.return %c : f32
}

// CHECK: sdfg.sdfg @test_remsi (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_remsi(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_srem, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.remsi %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_remui (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_remui(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_urem, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.remui %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_shli (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_shli(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_shl, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.shli %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_shrsi (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_shrsi(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_ashr, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.shrsi %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_shrui (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_shrui(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_lshr, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.shrui %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_sitofp (%[[ARG0:.*]]: i32) -> f32
sdfg.sdfg @test_sitofp(%a : i32) -> f32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (i32) -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : f32 -> f32
    // CHECK: sdfg.yield %[[VAR3]] : f32
    %b = arith.sitofp %a : i32 to f32
    // CHECK: sdfg.return %[[VAR0]] : f32
    sdfg.return %b : f32
}

// CHECK: sdfg.sdfg @test_subf (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> f32
sdfg.sdfg @test_subf(%a : f32, %b : f32) -> f32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : f32 -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet fp_sub, %[[VAR1]], %[[VAR2]] : (f32, f32) -> f32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : f32 -> f32
    // CHECK: sdfg.yield %[[VAR4]] : f32
    %c = arith.subf %a, %b : f32
    // CHECK: sdfg.return %[[VAR0]] : f32
    sdfg.return %c : f32
}

// CHECK: sdfg.sdfg @test_subi (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_subi(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_sub, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.subi %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}

// CHECK: sdfg.sdfg @test_truncf (%[[ARG0:.*]]: f32) -> f16
sdfg.sdfg @test_truncf(%a : f32) -> f16 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f16
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : f32 -> f32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (f32) -> f16
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : f16 -> f16
    // CHECK: sdfg.yield %[[VAR3]] : f16
    %b = arith.truncf %a : f32 to f16
    // CHECK: sdfg.return %[[VAR0]] : f16
    sdfg.return %b : f16
}

// CHECK: sdfg.sdfg @test_trunci (%[[ARG0:.*]]: i32) -> i16
sdfg.sdfg @test_trunci(%a : i32) -> i16 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i16
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (i32) -> i16
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : i16 -> i16
    // CHECK: sdfg.yield %[[VAR3]] : i16
    %b = arith.trunci %a : i32 to i16
    // CHECK: sdfg.return %[[VAR0]] : i16
    sdfg.return %b : i16
}

// CHECK: sdfg.sdfg @test_uitofp (%[[ARG0:.*]]: i32) -> f32
sdfg.sdfg @test_uitofp(%a : i32) -> f32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> f32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.tasklet assign, %[[VAR1]] : (i32) -> f32
    // CHECK: %[[VAR3:.*]] = sdfg.memlet %[[VAR2]] : f32 -> f32
    // CHECK: sdfg.yield %[[VAR3]] : f32
    %b = arith.uitofp %a : i32 to f32
    // CHECK: sdfg.return %[[VAR0]] : f32
    sdfg.return %b : f32
}

// CHECK: sdfg.sdfg @test_xori (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
sdfg.sdfg @test_xori(%a : i32, %b : i32) -> i32 {
    // CHECK: %[[VAR0:.*]] = sdfg.block -> i32
    // CHECK: %[[VAR1:.*]] = sdfg.memlet %[[ARG0]] : i32 -> i32
    // CHECK: %[[VAR2:.*]] = sdfg.memlet %[[ARG1]] : i32 -> i32
    // CHECK: %[[VAR3:.*]] = sdfg.tasklet int_xor, %[[VAR1]], %[[VAR2]] : (i32, i32) -> i32
    // CHECK: %[[VAR4:.*]] = sdfg.memlet %[[VAR3]] : i32 -> i32
    // CHECK: sdfg.yield %[[VAR4]] : i32
    %c = arith.xori %a, %b : i32
    // CHECK: sdfg.return %[[VAR0]] : i32
    sdfg.return %c : i32
}