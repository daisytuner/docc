#include <gtest/gtest.h>

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/passes/scheduler/loop_scheduling_pass.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/structured_control_flow/map.h>

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

void schedule_for_axelera(StructuredSDFG& sdfg) {
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);
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

# docc: SDFG arguments in order; "out" buffers are returned through A.output.
_docc_arguments = [("a", "in"), ("b", "in"), ("out", "out")]


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
                    r0 = A.array("float18x64")
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
    EXPECT_EQ(grid.schedule_type().value(), axelera::ScheduleType_Axelera_Datapath::value());
    auto& lanes = dynamic_cast<structured_control_flow::Map&>(grid.root().at(0));
    EXPECT_EQ(lanes.schedule_type().value(), axelera::ScheduleType_Axelera_Lane::value());
}

TEST(AxScriptTest, ConstantsAreRejected) {
    auto sdfg = elementwise(512, data_flow::TaskletCode::fp_mul, {"a"}, true);
    schedule_for_axelera(*sdfg);

    EXPECT_NE(error_of(*sdfg).find("constants is not supported"), std::string::npos) << error_of(*sdfg);
}

TEST(AxScriptTest, PartialPwordsAreRejected) {
    auto sdfg = elementwise(100, data_flow::TaskletCode::fp_add, {"a", "b"});
    schedule_for_axelera(*sdfg);

    EXPECT_NE(error_of(*sdfg).find("not scheduled for Axelera"), std::string::npos) << error_of(*sdfg);
}

TEST(AxScriptTest, ThirdInputStreamIsRejected) {
    auto sdfg = elementwise(512, data_flow::TaskletCode::fp_fma, {"a", "b", "c"});
    schedule_for_axelera(*sdfg);

    EXPECT_NE(error_of(*sdfg).find("third input stream"), std::string::npos) << error_of(*sdfg);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    auto context = plugins::Context::global_context();
    codegen::register_default_dispatchers();
    serializer::register_default_serializers();
    docc::target::axelera::register_plugin(context);
    return RUN_ALL_TESTS();
}
