#include "sdfg/codegen/code_generators/python_code_generator.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

using namespace sdfg;

namespace {

std::string generate_python(StructuredSDFG& sdfg) {
    analysis::AnalysisManager analysis_manager(sdfg);
    auto instrumentation_plan = codegen::InstrumentationPlan::none(sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(sdfg);
    codegen::PythonCodeGenerator generator(sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    return generator.function_definition() + "\n" + generator.main().str();
}

} // namespace

TEST(PythonCodeGeneratorTest, EmptyFunction) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto sdfg = builder.move();

    EXPECT_EQ(generate_python(*sdfg), "def sdfg_a():\n    with np.errstate(all=\"ignore\"):\n        pass\n");
}

TEST(PythonCodeGeneratorTest, VectorAdd) {
    builder::StructuredSDFGBuilder builder("vadd", FunctionType_CPU);
    types::Scalar f64(types::PrimitiveType::Double);
    types::Pointer ptr(f64);
    builder.add_container("A", ptr, true);
    builder.add_container("B", ptr, true);
    builder.add_container("C", ptr, true);
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));

    auto i = symbolic::symbol("i");
    auto& map = builder.add_map(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::symbol("N")),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& block = builder.add_block(map.root());
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, builder.add_access(block, "A"), tasklet, "_in1", {i}, ptr);
    builder.add_computational_memlet(block, builder.add_access(block, "B"), tasklet, "_in2", {i}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", builder.add_access(block, "C"), {i}, ptr);

    auto sdfg = builder.move();
    EXPECT_EQ(
        generate_python(*sdfg),
        "def vadd(A, B, C, N):\n"
        "    with np.errstate(all=\"ignore\"):\n"
        "        i = np.int64(0)\n"
        "        # Map\n"
        "        for i in range(0, N):\n"
        "            _in1 = np.float64((A)[i])\n"
        "            _in2 = np.float64((B)[i])\n"
        "\n"
        "            _out = _in1 + _in2\n"
        "\n"
        "            (C)[i] = _out\n"
    );
}

TEST(PythonCodeGeneratorTest, LocalsAreAllocatedWithNumpy) {
    builder::StructuredSDFGBuilder builder("locals", FunctionType_CPU);
    auto N = symbolic::symbol("N");
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);

    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer heap(
        types::StorageType::CPU_Heap(
            symbolic::mul(N, symbolic::integer(4)),
            types::StorageType::AllocationType::Managed,
            types::StorageType::AllocationType::Managed
        ),
        0,
        "",
        f32
    );
    builder.add_container("tmp", heap);
    builder.add_container("tile", types::Array(types::Array(f32, symbolic::integer(4)), symbolic::integer(8)));
    builder.add_container("view", types::Pointer(f32));

    auto sdfg = builder.move();
    auto code = generate_python(*sdfg);
    EXPECT_NE(code.find("tmp = np.zeros(_docc_idiv(4*N, 4), np.float32)\n"), std::string::npos) << code;
    EXPECT_NE(code.find("tile = np.zeros((8, 4,), np.float32)\n"), std::string::npos) << code;
    EXPECT_NE(code.find("view = None\n"), std::string::npos) << code;
}

TEST(PythonCodeGeneratorTest, PointerToArrayArgumentIsReshaped) {
    builder::StructuredSDFGBuilder builder("rows", FunctionType_CPU);
    types::Scalar f32(types::PrimitiveType::Float);
    builder.add_container("A", types::Pointer(types::Array(f32, symbolic::integer(16))), true);

    auto sdfg = builder.move();
    EXPECT_EQ(
        generate_python(*sdfg),
        "def rows(A):\n    A = A.reshape(-1, 16)\n    with np.errstate(all=\"ignore\"):\n        pass\n"
    );
}

TEST(PythonCodeGeneratorTest, CMathUsesNumpy) {
    builder::StructuredSDFGBuilder builder("cm", FunctionType_CPU);
    types::Scalar f32(types::PrimitiveType::Float);
    builder.add_container("x", f32, true);
    builder.add_container("y", f32);
    auto& block = builder.add_block(builder.subject().root());
    auto& node = builder.add_library_node<
        math::cmath::CMathNode>(block, DebugInfo(), math::cmath::CMathFunction::asin, types::PrimitiveType::Float);
    builder.add_computational_memlet(block, builder.add_access(block, "x"), node, "_in1", {}, f32);
    builder.add_computational_memlet(block, node, "_out", builder.add_access(block, "y"), {}, f32);

    auto sdfg = builder.move();
    auto code = generate_python(*sdfg);
    EXPECT_NE(code.find("_out = np.arcsin(_in1)\n"), std::string::npos) << code;
    EXPECT_NE(code.find("y = _out\n"), std::string::npos) << code;
}

TEST(PythonCodeGeneratorTest, KeywordConnectorsAreEscaped) {
    builder::StructuredSDFGBuilder builder("kw", FunctionType_CPU);
    types::Scalar f64(types::PrimitiveType::Double);
    builder.add_container("x", f64, true);
    builder.add_container("y", f64, true);
    auto& block = builder.add_block(builder.subject().root());
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out", {"in"});
    builder.add_computational_memlet(block, builder.add_access(block, "x"), tasklet, "in", {}, f64);
    builder.add_computational_memlet(block, tasklet, "out", builder.add_access(block, "y"), {}, f64);

    auto sdfg = builder.move();
    auto code = generate_python(*sdfg);
    EXPECT_NE(code.find("in_ = np.float64(x)\n"), std::string::npos) << code;
    EXPECT_NE(code.find("out = in_\n"), std::string::npos) << code;
    EXPECT_NE(code.find("y = out\n"), std::string::npos) << code;
}

TEST(PythonCodeGeneratorTest, RejectsLibraryNodes) {
    builder::StructuredSDFGBuilder builder("lib", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());
    builder.add_library_node<data_flow::BarrierLocalNode>(block, DebugInfo());

    auto sdfg = builder.move();
    EXPECT_THROW(generate_python(*sdfg), std::runtime_error);
}

TEST(PythonCodeGeneratorTest, RejectsParallelSchedules) {
    builder::StructuredSDFGBuilder builder("par", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    auto i = symbolic::symbol("i");
    builder.add_map(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType("CUDA", structured_control_flow::ScheduleTypeCategory::Offloader)
    );

    auto sdfg = builder.move();
    EXPECT_THROW(generate_python(*sdfg), std::runtime_error);
}

TEST(PythonCodeGeneratorTest, RejectsInstrumentation) {
    builder::StructuredSDFGBuilder builder("instr", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    auto i = symbolic::symbol("i");
    builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(8)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );

    auto sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*sdfg);
    auto instrumentation_plan = codegen::InstrumentationPlan::outermost_loops_plan(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    EXPECT_THROW(
        codegen::PythonCodeGenerator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan),
        std::runtime_error
    );
}
