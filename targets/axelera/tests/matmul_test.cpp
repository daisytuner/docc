#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/passes/expansion/library_node_expansion_pass.h>
#include <sdfg/passes/scheduler/loop_scheduling_pass.h>
#include <sdfg/passes/structured_control_flow/sequence_fusion.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/tiles/library_nodes/tile_copy_node.h>
#include <sdfg/tiles/transformations/local_storage.h>
#include <sdfg/tiles/transformations/software_pipelining.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/visitor/for_each.h>

#include "docc/target/axelera/axelera_scheduler.h"
#include "docc/target/axelera/axscript_code_generator.h"
#include "docc/target/axelera/matmul_node.h"
#include "docc/target/axelera/matmul_transforms.h"
#include "docc/target/axelera/schedule.h"
#include "docc/target/axelera/tile_target.h"

using namespace sdfg;

namespace {

struct MatMulNest {
    std::unique_ptr<StructuredSDFG> sdfg;
    /// The row loop of the MVM in the AI core, and the loop over its row tiles once tiled.
    structured_control_flow::StructuredLoop* rows = nullptr;
    structured_control_flow::StructuredLoop* tiles = nullptr;
};

/// c = a @ b for float a (m x k), b (k x n), as the Python frontend's `@` emits it: a GEMM node, expanded into a
/// zero-initialization nest and an accumulation nest.
MatMulNest float_matmul(int64_t m, int64_t n, int64_t k, const std::string& beta = "0.0") {
    builder::StructuredSDFGBuilder builder("mm", FunctionType_CPU);
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer f32_pointer(f32);
    for (auto name : {"a", "b", "c"}) {
        builder.add_container(name, f32_pointer, true);
    }
    auto& block = builder.add_block(builder.subject().root());
    auto& gemm = builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::s,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::No,
        symbolic::integer(m),
        symbolic::integer(n),
        symbolic::integer(k),
        symbolic::integer(k),
        symbolic::integer(n),
        symbolic::integer(n)
    );
    builder.add_computational_memlet(block, builder.add_access(block, "a"), gemm, "__A", {}, f32_pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "b"), gemm, "__B", {}, f32_pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "c"), gemm, "__C", {}, f32_pointer);
    builder.add_computational_memlet(block, builder.add_constant(block, "1.0", f32), gemm, "__alpha", {}, f32);
    builder.add_computational_memlet(block, builder.add_constant(block, beta, f32), gemm, "__beta", {}, f32);
    builder.add_return(builder.subject().root(), "");

    EXPECT_TRUE(passes::expansion::expand_single_math_node(builder, block, gemm).expanded);
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::SequenceFusion().run(builder, analysis_manager);
    return {builder.move()};
}

/// The nest accumulating the products, the last loop at the top level.
structured_control_flow::StructuredLoop& accumulation(StructuredSDFG& sdfg) {
    structured_control_flow::StructuredLoop* last = nullptr;
    for (size_t i = 0; i < sdfg.root().size(); ++i) {
        if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&sdfg.root().at(i))) {
            last = loop;
        }
    }
    if (!last) {
        throw std::runtime_error("no loop nest");
    }
    return *last;
}

/// The access feeding @p connector of the single MVM node under @p scope.
const data_flow::AccessNode& mvm_operand(structured_control_flow::ControlFlowNode& scope, const std::string& connector) {
    const data_flow::AccessNode* found = nullptr;
    visitor::for_each_block(scope, [&](structured_control_flow::Block& block) {
        for (auto* node : block.dataflow().library_nodes()) {
            if (!dynamic_cast<axelera::AxeleraMatMulNode*>(node)) {
                continue;
            }
            for (auto& memlet : block.dataflow().in_edges(*node)) {
                if (memlet.dst_conn() == connector) {
                    found = &static_cast<const data_flow::AccessNode&>(memlet.src());
                }
            }
        }
    });
    if (!found) {
        throw std::runtime_error("no MVM operand " + connector);
    }
    return *found;
}

size_t count_nodes(structured_control_flow::ControlFlowNode& scope, auto&& is) {
    size_t count = 0;
    visitor::for_each_block(scope, [&](structured_control_flow::Block& block) {
        for (auto* node : block.dataflow().library_nodes()) {
            count += is(*node) ? 1 : 0;
        }
    });
    return count;
}

void lower_to_mvm(MatMulNest& nest) {
    auto& sdfg = *nest.sdfg;
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);
    axelera::MvmEinsumTransform mvm(accumulation(sdfg));
    ASSERT_TRUE(mvm.can_be_applied(builder, analysis_manager));
    mvm.apply(builder, analysis_manager);
    nest.rows = mvm.row_loop();
    ASSERT_NE(nest.rows, nullptr);
}

/// The agent steps: run the matmul on the MVM, tile its rows, stage row tiles and the weights in L1, keep the
/// weights resident in the IMC.
void stage_matmul(MatMulNest& nest, int64_t tile_rows) {
    lower_to_mvm(nest);
    auto& sdfg = *nest.sdfg;
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& rows = *nest.rows;

    transformations::LoopTiling tiling(rows, tile_rows);
    ASSERT_TRUE(tiling.can_be_applied(builder, analysis_manager));
    tiling.apply(builder, analysis_manager);
    nest.tiles = tiling.outer_loop();
    auto& tiles = *nest.tiles;

    for (auto connector : {"A", "Y"}) {
        SCOPED_TRACE(connector);
        transformations::LocalStorage stage(rows, mvm_operand(rows, connector));
        ASSERT_TRUE(stage.can_be_applied(builder, analysis_manager));
        EXPECT_EQ(stage.storage_type(), axelera::StorageType_L1());
        stage.apply(builder, analysis_manager);
    }
    transformations::LocalStorage weights(tiles, mvm_operand(tiles, "B"));
    ASSERT_TRUE(weights.can_be_applied(builder, analysis_manager));
    weights.apply(builder, analysis_manager);

    axelera::ImcResidency residency(tiles);
    ASSERT_TRUE(residency.can_be_applied(builder, analysis_manager));
    residency.apply(builder, analysis_manager);
    EXPECT_EQ(sdfg.type(residency.imc_container()).storage_type(), axelera::StorageType_IMC());
}

/// Puts the quantization's maps and reductions on datapaths.
void schedule_datapaths(StructuredSDFG& sdfg) {
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);
    passes::scheduler::AxeleraScheduler scheduler;
    passes::scheduler::LoopSchedulingPass pass({&scheduler}, nullptr);
    pass.run(builder, analysis_manager);
}

std::string axscript_of(StructuredSDFG& sdfg) {
    schedule_datapaths(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);
    auto instrumentation_plan = codegen::InstrumentationPlan::none(sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(sdfg);
    axelera::AxScriptCodeGenerator generator(sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    std::ostringstream out;
    generator.emit_main_source(out, "");
    // Lets alto and the SDK interpreter check the generated modules outside the test.
    if (auto directory = std::getenv("DOCC_AXSCRIPT_DUMP")) {
        auto test = testing::UnitTest::GetInstance()->current_test_info()->name();
        std::ofstream(std::string(directory) + "/" + test + ".py") << out.str();
    }
    return out.str();
}

} // namespace

TEST(AxeleraMatMulTest, LowersFloatMatMulToQuantizedMvm) {
    auto nest = float_matmul(128, 128, 128);
    lower_to_mvm(nest);
    auto& sdfg = *nest.sdfg;
    sdfg.validate();

    // absmax and quantization of a and b, the MVM in one AI core, and the dequantization into c.
    std::vector<std::string> loops;
    for (size_t i = 0; i < sdfg.root().size(); ++i) {
        if (dynamic_cast<structured_control_flow::Reduce*>(&sdfg.root().at(i))) {
            loops.push_back("reduce");
        } else if (auto map = dynamic_cast<structured_control_flow::Map*>(&sdfg.root().at(i))) {
            loops.push_back(
                axelera::ScheduleType_Axelera::is(map->schedule_type(), axelera::AxeleraLevel::Core) ? "core" : "map"
            );
        }
    }
    EXPECT_EQ(loops, (std::vector<std::string>{"reduce", "map", "reduce", "map", "core", "map"}));
    EXPECT_EQ(
        count_nodes(
            sdfg.root(),
            [](auto& node) {
                return dynamic_cast<axelera::AxeleraMatMulNode*>(&node);
            }
        ),
        1
    );
    EXPECT_EQ(nest.rows->num_iterations()->__str__(), "128");
}

TEST(AxeleraMatMulTest, RequiresZeroInitializedResult) {
    auto nest = float_matmul(128, 128, 128, "1.0");
    builder::StructuredSDFGBuilder builder(*nest.sdfg);
    analysis::AnalysisManager analysis_manager(*nest.sdfg);
    axelera::MvmEinsumTransform mvm(accumulation(*nest.sdfg));
    EXPECT_FALSE(mvm.can_be_applied(builder, analysis_manager));
}

TEST(AxeleraMatMulTest, StagesTilesAndWeights) {
    auto nest = float_matmul(128, 128, 128);
    stage_matmul(nest, 32);
    auto& sdfg = *nest.sdfg;
    auto is_copy = [](auto& node) {
        return dynamic_cast<tiles::TileCopyNode*>(&node) != nullptr;
    };
    auto is_load = [](auto& node) {
        return dynamic_cast<axelera::AxeleraImcLoadNode*>(&node) != nullptr;
    };
    auto is_matmul = [](auto& node) {
        return dynamic_cast<axelera::AxeleraMatMulNode*>(&node) != nullptr;
    };
    EXPECT_EQ(count_nodes(sdfg.root(), is_copy), 3);
    EXPECT_EQ(count_nodes(sdfg.root(), is_load), 1);
    EXPECT_EQ(count_nodes(sdfg.root(), is_matmul), 1);
    sdfg.validate();
}

TEST(AxeleraMatMulTest, PipelinesRowTiles) {
    auto nest = float_matmul(128, 128, 128);
    stage_matmul(nest, 32);
    auto& sdfg = *nest.sdfg;
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);

    transformations::SoftwarePipelining pipelining(*nest.tiles, 2);
    ASSERT_TRUE(pipelining.can_be_applied(builder, analysis_manager));
    pipelining.apply(builder, analysis_manager);

    // Only the copied-in A tile is double-buffered; the matmul's output tile stays single.
    auto& a_tile = static_cast<const types::Array&>(sdfg.type(mvm_operand(*nest.rows, "A").data()));
    EXPECT_TRUE(symbolic::eq(a_tile.num_elements(), symbolic::integer(2)));
    auto& y_tile = static_cast<const types::Array&>(sdfg.type(mvm_operand(*nest.rows, "Y").data()));
    EXPECT_TRUE(symbolic::eq(y_tile.num_elements(), symbolic::integer(32)));
    sdfg.validate();
}

TEST(AxeleraMatMulTest, GeneratesPipelinedMatMul) {
    auto nest = float_matmul(128, 128, 128);
    stage_matmul(nest, 32);
    auto& sdfg = *nest.sdfg;
    {
        builder::StructuredSDFGBuilder builder(sdfg);
        analysis::AnalysisManager analysis_manager(sdfg);
        transformations::SoftwarePipelining pipelining(*nest.tiles, 2);
        ASSERT_TRUE(pipelining.can_be_applied(builder, analysis_manager));
        pipelining.apply(builder, analysis_manager);
    }
    // The next row tile is copied into the other stage while the MVM works on this one. AxScript waits on a copy in
    // the iteration that issued it, so the wait for the prefetch closes the iteration, and the prologue's copy is
    // waited on before the loop.
    auto code = axscript_of(sdfg);
    EXPECT_NE(
        code.find(R"(        with A.AICore(core=0):
            axelera_imc___daisy_local_storage_b_int8000 = A.alloc_imc((2, 2))
            daisy_local_storage_a_int800 = A.alloc_buffer((2, 32, 2), "int8x64", scope=A.L1)
            daisy_local_storage_b_int800 = A.alloc_buffer((128, 2), "int8x64", scope=A.L1)
            daisy_local_storage_c_products00 = A.alloc_buffer((32, 2), "float32x64", scope=A.L1)
            atomic.copy(b_int80, scope=A.L1, output=daisy_local_storage_b_int800)
            atomic.load_imc(buffer=daisy_local_storage_b_int800, imc=axelera_imc___daisy_local_storage_b_int8000)
            atomic.copy(a_int80[A.Ramp(0, 1, 32), A.Ramp(0, 1, 2)], scope=A.L1, output=daisy_local_storage_a_int800[0, A.Ramp(0, 1, 32), A.Ramp(0, 1, 2)], id="daisy_local_storage_a_int800_0")
            A.waits("daisy_local_storage_a_int800_0")
            for mvm_row0_tile0 in A.serial(4):
                atomic.copy(a_int80[A.Ramp(A.min(96, 32 + 32*mvm_row0_tile0), 1, 32), A.Ramp(0, 1, 2)], scope=A.L1, output=daisy_local_storage_a_int800[((1 + mvm_row0_tile0) % 2), A.Ramp(0, 1, 32), A.Ramp(0, 1, 2)], id="daisy_local_storage_a_int800_1", where=32 + 32*mvm_row0_tile0 < 128)
                atomic.matmul(matrix_ifd=daisy_local_storage_a_int800[(mvm_row0_tile0 % 2), A.Ramp(0, 1, 32), A.Ramp(0, 1, 2)], matrix_imc=axelera_imc___daisy_local_storage_b_int8000, output=daisy_local_storage_c_products00)
                atomic.copy(daisy_local_storage_c_products00, scope=A.DDR, output=c_products0[A.Ramp(32*mvm_row0_tile0, 1, 32), A.Ramp(0, 1, 2)])
                if 32 + 32*mvm_row0_tile0 < 128:
                    A.waits("daisy_local_storage_a_int800_1")
)"),
        std::string::npos
    ) << code;
}

TEST(AxeleraMatMulTest, GeneratesQuantizedMatMul) {
    auto nest = float_matmul(128, 128, 128);
    stage_matmul(nest, 32);
    // Per tensor: absmax in chunks of 64 PWORDs and their combine, quantization to an int8 matrix; then the MVM in
    // the AI core and the dequantization into c.
    EXPECT_EQ(
        axscript_of(*nest.sdfg),
        R"(import axelera.compiler.alto.script as A
from axelera.compiler.alto.axol import atomic

# docc: SDFG arguments in order with their element counts and PWORD buffer shapes ("out" buffers are returned through A.output) and AI cores used.
_docc_arguments = [("a", "in", 16384, (256,)), ("b", "in", 16384, (256,)), ("c", "out", 16384, (256,))]
_docc_num_cores = 1
# docc: the module manages local memory itself, which alto's tiling injection would redo.
_docc_tiling_injection = False
_docc_c0_lanes = [1e-30] * 64
_docc_c1_lanes = [127.0] * 64
_docc_c2_lanes = [6.2000124000248e-05] * 64


@A.ir_module
class mm:
    @A.prim_func
    def main(a: A.Buffer((256,), "float32x64", scope=A.DDR), b: A.Buffer((256,), "float32x64", scope=A.DDR)):
        c = A.alloc_buffer((256,), "float32x64", scope=A.DDR)
        a_int80 = A.alloc_buffer((128, 2), "int8x64", scope=A.DDR)
        b_int80 = A.alloc_buffer((128, 2), "int8x64", scope=A.DDR)
        c_products0 = A.alloc_buffer((128, 2), "float32x64", scope=A.DDR)
        a_absmax0 = A.alloc_buffer((1,), "float32x64", scope=A.DDR)
        b_absmax0 = A.alloc_buffer((1,), "float32x64", scope=A.DDR)
        _docc_c0 = A.alloc_constant(_docc_c0_lanes, "float32x64", scope=A.DDR)
        _docc_c1 = A.alloc_constant(_docc_c1_lanes, "float32x64", scope=A.DDR)
        _docc_c2 = A.alloc_constant(_docc_c2_lanes, "float32x64", scope=A.DDR)
        a_absmax_acc0_partials = A.alloc_buffer((4,), "float32x64", scope=A.L2)
        for a_e0_tile0_chunk in A.grid(4):
            with A.Datapath():
                s0_0 = A.decl_stream("float32x64")
                s0_out = A.decl_stream("float32x64")
                with A.IFD(num=0):
                    for a_e0_tile0 in A.serial(64):
                        s0_0.push(a[a_e0_tile0 + 64*a_e0_tile0_chunk])
                with A.DPU():
                    r0 = A.array("float18x64", size=4)
                    r0[0] = A.cast(s0_0.pop(), "float18x64")
                    r0[1] = A.dpu.ops.neg(r0[0])
                    r0[2] = A.max(r0[0], r0[1])
                    r0[3] = r0[2]
                    for a_e0_tile0 in A.serial(63):
                        r0[0] = A.cast(s0_0.pop(), "float18x64")
                        r0[1] = A.dpu.ops.neg(r0[0])
                        r0[2] = A.max(r0[0], r0[1])
                        r0[3] = A.max(r0[3], r0[2])
                    s0_out.push(r0[3])
                with A.ODR():
                    a_absmax_acc0_partials[a_e0_tile0_chunk] = s0_out.pop()
        with A.Datapath():
            s1_0 = A.decl_stream("float32x64")
            s1_out = A.decl_stream("float32x64")
            with A.IFD(num=0):
                for c1 in A.serial(4):
                    s1_0.push(a_absmax_acc0_partials[c1])
            with A.DPU():
                r1 = A.array("float18x64", size=2)
                r1[0] = A.cast(s1_0.pop(), "float18x64")
                for c1 in A.serial(3):
                    r1[1] = A.cast(s1_0.pop(), "float18x64")
                    r1[0] = A.max(r1[0], r1[1])
                r1[0] = A.dpu.ops.max_reduce(r1[0])
                s1_out.push(r1[0])
            with A.ODR():
                a_absmax0[0] = s1_out.pop()
        for a_q0_tile0_row, a_q0_tile0_col in A.grid(128, 2):
            with A.Datapath():
                s2_0 = A.decl_stream("float32x64")
                s2_out = A.decl_stream("int8x64")
                with A.IFD(num=0):
                    s2_0.push(a[a_q0_tile0_col + 2*a_q0_tile0_row])
                with A.DPU():
                    r2 = A.array("float18x64", size=10)
                    r2[0] = A.cast(a_absmax0[0], "float18x64")
                    r2[1] = A.cast(_docc_c0[0], "float18x64")
                    r2[2] = A.max(r2[0], r2[1])
                    r2[3] = A.cast(s2_0.pop(), "float18x64")
                    r2[4] = A.dpu.ops.recip(r2[2])
                    r2[5] = r2[3] * r2[4]
                    r2[6] = A.cast(_docc_c1[0], "float18x64")
                    r2[7] = r2[5] * r2[6]
                    r2[8] = r2[7]
                    r2[9] = r2[8]
                    s2_out.push(A.cast(r2[9], "int8x64"))
                with A.ODR():
                    a_int80[a_q0_tile0_row, a_q0_tile0_col] = s2_out.pop()
        b_absmax_acc0_partials = A.alloc_buffer((4,), "float32x64", scope=A.L2)
        for b_e0_tile0_chunk in A.grid(4):
            with A.Datapath():
                s3_0 = A.decl_stream("float32x64")
                s3_out = A.decl_stream("float32x64")
                with A.IFD(num=0):
                    for b_e0_tile0 in A.serial(64):
                        s3_0.push(b[b_e0_tile0 + 64*b_e0_tile0_chunk])
                with A.DPU():
                    r3 = A.array("float18x64", size=4)
                    r3[0] = A.cast(s3_0.pop(), "float18x64")
                    r3[1] = A.dpu.ops.neg(r3[0])
                    r3[2] = A.max(r3[0], r3[1])
                    r3[3] = r3[2]
                    for b_e0_tile0 in A.serial(63):
                        r3[0] = A.cast(s3_0.pop(), "float18x64")
                        r3[1] = A.dpu.ops.neg(r3[0])
                        r3[2] = A.max(r3[0], r3[1])
                        r3[3] = A.max(r3[3], r3[2])
                    s3_out.push(r3[3])
                with A.ODR():
                    b_absmax_acc0_partials[b_e0_tile0_chunk] = s3_out.pop()
        with A.Datapath():
            s4_0 = A.decl_stream("float32x64")
            s4_out = A.decl_stream("float32x64")
            with A.IFD(num=0):
                for c4 in A.serial(4):
                    s4_0.push(b_absmax_acc0_partials[c4])
            with A.DPU():
                r4 = A.array("float18x64", size=2)
                r4[0] = A.cast(s4_0.pop(), "float18x64")
                for c4 in A.serial(3):
                    r4[1] = A.cast(s4_0.pop(), "float18x64")
                    r4[0] = A.max(r4[0], r4[1])
                r4[0] = A.dpu.ops.max_reduce(r4[0])
                s4_out.push(r4[0])
            with A.ODR():
                b_absmax0[0] = s4_out.pop()
        for b_q0_tile0_row, b_q0_tile0_col in A.grid(128, 2):
            with A.Datapath():
                s5_0 = A.decl_stream("float32x64")
                s5_out = A.decl_stream("int8x64")
                with A.IFD(num=0):
                    s5_0.push(b[b_q0_tile0_col + 2*b_q0_tile0_row])
                with A.DPU():
                    r5 = A.array("float18x64", size=10)
                    r5[0] = A.cast(b_absmax0[0], "float18x64")
                    r5[1] = A.cast(_docc_c0[0], "float18x64")
                    r5[2] = A.max(r5[0], r5[1])
                    r5[3] = A.cast(s5_0.pop(), "float18x64")
                    r5[4] = A.dpu.ops.recip(r5[2])
                    r5[5] = r5[3] * r5[4]
                    r5[6] = A.cast(_docc_c1[0], "float18x64")
                    r5[7] = r5[5] * r5[6]
                    r5[8] = r5[7]
                    r5[9] = r5[8]
                    s5_out.push(A.cast(r5[9], "int8x64"))
                with A.ODR():
                    b_int80[b_q0_tile0_row, b_q0_tile0_col] = s5_out.pop()
        with A.AICore(core=0):
            axelera_imc___daisy_local_storage_b_int8000 = A.alloc_imc((2, 2))
            daisy_local_storage_a_int800 = A.alloc_buffer((32, 2), "int8x64", scope=A.L1)
            daisy_local_storage_b_int800 = A.alloc_buffer((128, 2), "int8x64", scope=A.L1)
            daisy_local_storage_c_products00 = A.alloc_buffer((32, 2), "float32x64", scope=A.L1)
            atomic.copy(b_int80, scope=A.L1, output=daisy_local_storage_b_int800)
            atomic.load_imc(buffer=daisy_local_storage_b_int800, imc=axelera_imc___daisy_local_storage_b_int8000)
            for mvm_row0_tile0 in A.serial(4):
                atomic.copy(a_int80[A.Ramp(32*mvm_row0_tile0, 1, 32), A.Ramp(0, 1, 2)], scope=A.L1, output=daisy_local_storage_a_int800)
                atomic.matmul(matrix_ifd=daisy_local_storage_a_int800, matrix_imc=axelera_imc___daisy_local_storage_b_int8000, output=daisy_local_storage_c_products00)
                atomic.copy(daisy_local_storage_c_products00, scope=A.DDR, output=c_products0[A.Ramp(32*mvm_row0_tile0, 1, 32), A.Ramp(0, 1, 2)])
        for c_e0_tile0_row, c_e0_tile0_col in A.grid(128, 2):
            with A.Datapath():
                s6_0 = A.decl_stream("float32x64")
                s6_out = A.decl_stream("float32x64")
                with A.IFD(num=0):
                    s6_0.push(c_products0[c_e0_tile0_row, c_e0_tile0_col])
                with A.DPU():
                    r6 = A.array("float18x64", size=7)
                    r6[0] = A.cast(a_absmax0[0], "float18x64")
                    r6[1] = A.cast(b_absmax0[0], "float18x64")
                    r6[2] = r6[0] * r6[1]
                    r6[3] = A.cast(_docc_c2[0], "float18x64")
                    r6[4] = r6[2] * r6[3]
                    r6[5] = A.cast(s6_0.pop(), "float18x64")
                    r6[6] = r6[5] * r6[4]
                    s6_out.push(r6[6])
                with A.ODR():
                    c[c_e0_tile0_col + 2*c_e0_tile0_row] = s6_out.pop()
        A.output(c)
)"
    );
}
