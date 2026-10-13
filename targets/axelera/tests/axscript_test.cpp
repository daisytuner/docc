#include <gtest/gtest.h>

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/cmath/cmath_node.h>
#include <sdfg/passes/scheduler/loop_scheduling_pass.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/reduce.h>

#include "docc/target/axelera/axelera_scheduler.h"
#include "docc/target/axelera/axscript_code_generator.h"
#include "docc/target/axelera/plugin.h"
#include "docc/target/axelera/schedule.h"

using namespace sdfg;

namespace {

/// out[i] = combine(a[i], b[i]) for i < n, as the Python frontend emits elementwise array operations.
std::unique_ptr<StructuredSDFG> elementwise(
    int64_t n, data_flow::TaskletCode code, const std::vector<std::string>& inputs, bool constant_operand = false
) {
    builder::StructuredSDFGBuilder builder("ew", FunctionType_CPU);
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer ptr(f32);
    for (auto& input : inputs) {
        builder.add_container(input, ptr, true);
    }
    builder.add_container("out", ptr, true);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));

    auto i = symbolic::symbol("i");
    auto& map = builder.add_map(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(n)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block = builder.add_block(map.root());
    std::vector<std::string> connectors;
    for (size_t k = 0; k < inputs.size() + (constant_operand ? 1 : 0); ++k) {
        connectors.push_back("_in" + std::to_string(k + 1));
    }
    auto& tasklet = builder.add_tasklet(block, code, "_out", connectors);
    for (size_t k = 0; k < inputs.size(); ++k) {
        builder.add_computational_memlet(block, builder.add_access(block, inputs[k]), tasklet, connectors[k], {i}, ptr);
    }
    if (constant_operand) {
        builder
            .add_computational_memlet(block, builder.add_constant(block, "2.0", f32), tasklet, connectors.back(), {}, f32);
    }
    builder.add_computational_memlet(block, tasklet, "_out", builder.add_access(block, "out"), {i}, ptr);
    builder.add_return(builder.subject().root(), "");
    return builder.move();
}

/// out[0] = reduction of a[k] for k < n starting from @p start, as the Python frontend emits np.sum/max/min.
std::unique_ptr<StructuredSDFG>
reduction(int64_t n, structured_control_flow::ReductionOperation operation, const std::string& start) {
    using structured_control_flow::ReductionOperation;
    builder::StructuredSDFGBuilder builder("red", FunctionType_CPU);
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer ptr(f32);
    builder.add_container("a", ptr, true);
    builder.add_container("out", ptr, true);
    builder.add_container("acc", f32);
    builder.add_container("k", types::Scalar(types::PrimitiveType::Int64));
    auto& root = builder.subject().root();

    auto& init = builder.add_block(root);
    auto& set = builder.add_tasklet(init, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(init, builder.add_constant(init, start, f32), set, "_in", {}, f32);
    builder.add_computational_memlet(init, set, "_out", builder.add_access(init, "acc"), {}, f32);

    auto k = symbolic::symbol("k");
    auto& loop = builder.add_reduce(
        root,
        k,
        symbolic::Lt(k, symbolic::integer(n)),
        symbolic::integer(0),
        symbolic::add(k, symbolic::integer(1)),
        {{operation, "acc"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = builder.add_block(loop.root());
    data_flow::CodeNode* combine;
    if (operation == ReductionOperation::Add) {
        combine = &builder.add_tasklet(body, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    } else {
        auto function = operation == ReductionOperation::Max ? math::cmath::CMathFunction::fmax
                                                             : math::cmath::CMathFunction::fmin;
        combine =
            &builder.add_library_node<math::cmath::CMathNode>(body, DebugInfo(), function, types::PrimitiveType::Float);
    }
    builder.add_computational_memlet(body, builder.add_access(body, "a"), *combine, "_in1", {k}, ptr);
    builder.add_computational_memlet(body, builder.add_access(body, "acc"), *combine, "_in2", {}, f32);
    builder.add_computational_memlet(body, *combine, "_out", builder.add_access(body, "acc"), {}, f32);

    auto& writeback = builder.add_block(root);
    auto& store = builder.add_tasklet(writeback, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(writeback, builder.add_access(writeback, "acc"), store, "_in", {}, f32);
    builder
        .add_computational_memlet(writeback, store, "_out", builder.add_access(writeback, "out"), {symbolic::zero()}, ptr);
    builder.add_return(root, "");
    return builder.move();
}

void schedule_for_axelera(StructuredSDFG& sdfg, int64_t cores = 1) {
    builder::StructuredSDFGBuilder builder(sdfg);
    Options options;
    options.set(std::string(axelera::NUM_CORES.key), cores);
    analysis::AnalysisManager analysis_manager(sdfg, options);
    passes::scheduler::AxeleraScheduler scheduler;
    passes::scheduler::LoopSchedulingPass pass({&scheduler}, nullptr);
    pass.run(builder, analysis_manager);
}

std::string generate_axscript(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto instrumentation_plan = codegen::InstrumentationPlan::none(sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(sdfg);
    axelera::AxScriptCodeGenerator generator(sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    std::ostringstream out;
    generator.emit_main_source(out, "");
    return out.str();
}

std::string error_of(StructuredSDFG& sdfg) {
    try {
        generate_axscript(sdfg);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "";
}

} // namespace

TEST(AxScriptTest, VectorAdd) {
    auto sdfg = elementwise(512, data_flow::TaskletCode::fp_add, {"a", "b"});
    schedule_for_axelera(*sdfg);

    EXPECT_EQ(
        generate_axscript(*sdfg),
        R"(import axelera.compiler.alto.script as A

# docc: SDFG arguments in order with their element counts ("out" buffers are returned through A.output) and AI cores used.
_docc_arguments = [("a", "in", 512), ("b", "in", 512), ("out", "out", 512)]
_docc_num_cores = 1


@A.ir_module
class ew:
    @A.prim_func
    def main(a: A.Buffer((8,), "float32x64", scope=A.DDR), b: A.Buffer((8,), "float32x64", scope=A.DDR)):
        out = A.alloc_buffer((8,), "float32x64", scope=A.DDR)
        for i_tile0 in A.grid(8):
            with A.Datapath():
                s0_0 = A.decl_stream("float32x64")
                s0_1 = A.decl_stream("float32x64")
                s0_out = A.decl_stream("float32x64")
                with A.IFD(num=0):
                    s0_0.push(a[i_tile0])
                with A.IFD(num=1):
                    s0_1.push(b[i_tile0])
                with A.DPU():
                    r0 = A.array("float18x64", size=3)
                    r0[0] = A.cast(s0_0.pop(), "float18x64")
                    r0[1] = A.cast(s0_1.pop(), "float18x64")
                    r0[2] = r0[0] + r0[1]
                    s0_out.push(r0[2])
                with A.ODR():
                    out[i_tile0] = s0_out.pop()
        A.output(out)
)"
    );
}

TEST(AxScriptTest, SchedulerTilesByPword) {
    auto sdfg = elementwise(512, data_flow::TaskletCode::fp_mul, {"a", "b"});
    schedule_for_axelera(*sdfg);

    auto& grid = dynamic_cast<structured_control_flow::Map&>(sdfg->root().at(0));
    EXPECT_TRUE(axelera::ScheduleType_Axelera::is(grid.schedule_type(), axelera::AxeleraLevel::Stream));
    auto& lanes = dynamic_cast<structured_control_flow::Map&>(grid.root().at(0));
    EXPECT_TRUE(axelera::ScheduleType_Axelera::is(lanes.schedule_type(), axelera::AxeleraLevel::Lane));
}

TEST(AxScriptTest, SchedulerSplitsPwordsOverCores) {
    auto sdfg = elementwise(512, data_flow::TaskletCode::fp_mul, {"a", "b"});
    schedule_for_axelera(*sdfg, 2);

    auto& cores = dynamic_cast<structured_control_flow::Map&>(sdfg->root().at(0));
    EXPECT_TRUE(axelera::ScheduleType_Axelera::is(cores.schedule_type(), axelera::AxeleraLevel::Core));
    EXPECT_EQ(axelera::ScheduleType_Axelera::num_cores(cores.schedule_type()), 2);
    auto& grid = dynamic_cast<structured_control_flow::Map&>(cores.root().at(0));
    EXPECT_TRUE(axelera::ScheduleType_Axelera::is(grid.schedule_type(), axelera::AxeleraLevel::Stream));
    auto& lanes = dynamic_cast<structured_control_flow::Map&>(grid.root().at(0));
    EXPECT_TRUE(axelera::ScheduleType_Axelera::is(lanes.schedule_type(), axelera::AxeleraLevel::Lane));
}

TEST(AxScriptTest, VectorAddOnTwoCores) {
    auto sdfg = elementwise(512, data_flow::TaskletCode::fp_add, {"a", "b"});
    schedule_for_axelera(*sdfg, 2);

    auto datapath = [](const std::string& id, const std::string& offset) {
        return R"(            for i_tile1 in A.grid(4):
                with A.Datapath():
                    s)" +
               id + R"(_0 = A.decl_stream("float32x64")
                    s)" +
               id + R"(_1 = A.decl_stream("float32x64")
                    s)" +
               id + R"(_out = A.decl_stream("float32x64")
                    with A.IFD(num=0):
                        s)" +
               id + "_0.push(a[" + offset + R"(i_tile1])
                    with A.IFD(num=1):
                        s)" +
               id + "_1.push(b[" + offset + R"(i_tile1])
                    with A.DPU():
                        r)" +
               id + R"( = A.array("float18x64", size=3)
                        r)" +
               id + "[0] = A.cast(s" + id + R"(_0.pop(), "float18x64")
                        r)" +
               id + "[1] = A.cast(s" + id + R"(_1.pop(), "float18x64")
                        r)" +
               id + "[2] = r" + id + "[0] + r" + id + R"([1]
                        s)" +
               id + "_out.push(r" + id + R"([2])
                    with A.ODR():
                        out[)" +
               offset + "i_tile1] = s" + id + R"(_out.pop()
)";
    };
    EXPECT_EQ(
        generate_axscript(*sdfg),
        R"(import axelera.compiler.alto.script as A

# docc: SDFG arguments in order with their element counts ("out" buffers are returned through A.output) and AI cores used.
_docc_arguments = [("a", "in", 512), ("b", "in", 512), ("out", "out", 512)]
_docc_num_cores = 2


@A.ir_module
class ew:
    @A.prim_func
    def main(a: A.Buffer((8,), "float32x64", scope=A.DDR), b: A.Buffer((8,), "float32x64", scope=A.DDR)):
        out = A.alloc_buffer((8,), "float32x64", scope=A.DDR)
        with A.AICore(core=0):
)" + datapath("0", "") +
            R"(            A.barrier(0, 1)
        with A.AICore(core=1):
)" + datapath("1", "4 + ") +
            R"(            A.barrier(0, 1)
        A.output(out)
)"
    );
}

TEST(AxScriptTest, UnevenCoreSplitIsRejected) {
    auto sdfg = elementwise(512, data_flow::TaskletCode::fp_add, {"a", "b"});
    schedule_for_axelera(*sdfg, 3);

    Options options;
    options.set(std::string(axelera::NUM_CORES.key), int64_t{3});
    analysis::AnalysisManager analysis_manager(*sdfg, options);
    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    axelera::AxScriptCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    try {
        generator.generate();
        FAIL() << "expected the uneven split to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("8 PWORDs do not split"), std::string::npos) << e.what();
    }
}

TEST(AxScriptTest, ConstantsAreRejected) {
    auto sdfg = elementwise(512, data_flow::TaskletCode::fp_mul, {"a"}, true);
    schedule_for_axelera(*sdfg);

    EXPECT_NE(error_of(*sdfg).find("constants is not supported"), std::string::npos) << error_of(*sdfg);
}

TEST(AxScriptTest, PartialPwordsAreRejected) {
    auto sdfg = elementwise(100, data_flow::TaskletCode::fp_add, {"a", "b"});
    schedule_for_axelera(*sdfg);

    EXPECT_NE(error_of(*sdfg).find("not whole PWORDs"), std::string::npos) << error_of(*sdfg);
}

TEST(AxScriptTest, ThirdInputStreamIsRejected) {
    auto sdfg = elementwise(512, data_flow::TaskletCode::fp_fma, {"a", "b", "c"});
    schedule_for_axelera(*sdfg);

    EXPECT_NE(error_of(*sdfg).find("third input stream"), std::string::npos) << error_of(*sdfg);
}

TEST(AxScriptTest, SumReduction) {
    auto sdfg = reduction(512, structured_control_flow::ReductionOperation::Add, "0.0");
    schedule_for_axelera(*sdfg);

    EXPECT_EQ(
        generate_axscript(*sdfg),
        R"(import axelera.compiler.alto.script as A

# docc: SDFG arguments in order with their element counts ("out" buffers are returned through A.output) and AI cores used.
_docc_arguments = [("a", "in", 512), ("out", "out", 1)]
_docc_num_cores = 1


@A.ir_module
class red:
    @A.prim_func
    def main(a: A.Buffer((8,), "float32x64", scope=A.DDR)):
        out = A.alloc_buffer((1,), "float32x64", scope=A.DDR)
        with A.Datapath():
            s0_0 = A.decl_stream("float32x64")
            s0_out = A.decl_stream("float32x64")
            with A.IFD(num=0):
                for k_tile0 in A.serial(8):
                    s0_0.push(a[k_tile0])
            with A.DPU():
                r0 = A.array("float18x64", size=2)
                r0[0] = A.cast(s0_0.pop(), "float18x64")
                r0[1] = r0[0]
                for k_tile0 in A.serial(7):
                    r0[0] = A.cast(s0_0.pop(), "float18x64")
                    r0[1] = r0[1] + r0[0]
                r0[1] = A.dpu.ops.add_reduce(r0[1])
                s0_out.push(r0[1])
            with A.ODR():
                out[0] = s0_out.pop()
        A.output(out)
)"
    );
}

TEST(AxScriptTest, MaxReductionOnTwoCores) {
    auto sdfg = reduction(512, structured_control_flow::ReductionOperation::Max, "-inf");
    schedule_for_axelera(*sdfg, 2);

    auto& cores = dynamic_cast<structured_control_flow::Reduce&>(sdfg->root().at(1));
    EXPECT_TRUE(axelera::ScheduleType_Axelera::is(cores.schedule_type(), axelera::AxeleraLevel::Core));

    auto code = generate_axscript(*sdfg);
    std::vector<std::string> in_order = {
        "_docc_num_cores = 2",
        "        acc_partials = A.alloc_buffer((2,), \"float32x64\", scope=A.L2)\n"
        "        with A.AICore(core=0):\n",
        "                    for k_tile1 in A.serial(4):\n"
        "                        s0_0.push(a[k_tile1])\n",
        "                        r0[1] = A.max(r0[1], r0[0])\n",
        "                    s0_out.push(r0[1])\n"
        "                with A.ODR():\n"
        "                    acc_partials[0] = s0_out.pop()\n"
        "            A.barrier(0, 1)\n"
        "        with A.AICore(core=1):\n",
        "                        s1_0.push(a[4 + k_tile1])\n",
        "                    acc_partials[1] = s1_out.pop()\n"
        "            A.barrier(0, 1)\n"
        "            with A.Datapath():\n",
        "                        s2_0.push(acc_partials[c2])\n",
        "                    r2[0] = A.dpu.ops.max_reduce(r2[0])\n",
        "                    out[0] = s2_out.pop()\n"
        "        A.output(out)\n",
    };
    size_t position = 0;
    for (auto& part : in_order) {
        auto found = code.find(part, position);
        ASSERT_NE(found, std::string::npos) << "missing in order: " << part << "\n" << code;
        position = found + part.size();
    }
    EXPECT_EQ(code.find("max_reduce"), code.rfind("max_reduce")) << "only the combine reduces lanes\n" << code;
}

TEST(AxScriptTest, MinReductionUsesMin) {
    auto sdfg = reduction(128, structured_control_flow::ReductionOperation::Min, "inf");
    schedule_for_axelera(*sdfg);

    auto code = generate_axscript(*sdfg);
    EXPECT_NE(code.find("r0[1] = A.min(r0[1], r0[0])"), std::string::npos) << code;
    EXPECT_NE(code.find("r0[1] = A.dpu.ops.min_reduce(r0[1])"), std::string::npos) << code;
}

TEST(AxScriptTest, NonIdentityStartIsRejected) {
    auto sdfg = reduction(512, structured_control_flow::ReductionOperation::Add, "1.0");
    schedule_for_axelera(*sdfg);

    EXPECT_NE(error_of(*sdfg).find("not the reduction's identity"), std::string::npos) << error_of(*sdfg);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    auto context = plugins::Context::global_context();
    codegen::register_default_dispatchers();
    serializer::register_default_serializers();
    docc::target::axelera::register_plugin(context);
    return RUN_ALL_TESTS();
}
