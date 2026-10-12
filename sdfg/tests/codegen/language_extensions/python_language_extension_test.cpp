#include "sdfg/codegen/language_extensions/python_language_extension.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"

using namespace sdfg;

namespace {

std::string dispatch_python(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node) {
    analysis::AnalysisManager analysis_manager(sdfg);
    codegen::PythonLanguageExtension language_extension(sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(sdfg);
    auto dispatcher =
        codegen::create_dispatcher(language_extension, sdfg, analysis_manager, node, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher->dispatch(main_stream, globals_stream, library_factory);
    return main_stream.str();
}

} // namespace

TEST(PythonLanguageExtensionTest, PrimitiveTypes) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    codegen::PythonLanguageExtension python(builder.subject());

    EXPECT_EQ(python.primitive_type(types::PrimitiveType::Bool), "np.bool_");
    EXPECT_EQ(python.primitive_type(types::PrimitiveType::Int32), "np.int32");
    EXPECT_EQ(python.primitive_type(types::PrimitiveType::UInt64), "np.uint64");
    EXPECT_EQ(python.primitive_type(types::PrimitiveType::Float), "np.float32");
    EXPECT_EQ(python.primitive_type(types::PrimitiveType::Double), "np.float64");
    EXPECT_THROW(python.primitive_type(types::PrimitiveType::Int128), std::runtime_error);
}

TEST(PythonLanguageExtensionTest, ConstantsBecomeTypedNumpyScalars) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());
    codegen::PythonLanguageExtension python(builder.subject());

    auto literal = [&](const std::string& data, types::PrimitiveType prim) {
        return python.access_node(builder.add_constant(block, data, types::Scalar(prim)));
    };
    EXPECT_EQ(literal("0x10", types::PrimitiveType::Int32), "np.int32(16)");
    EXPECT_EQ(literal("42ll", types::PrimitiveType::Int64), "np.int64(42)");
    EXPECT_EQ(literal("-1", types::PrimitiveType::UInt32), "np.uint32(4294967295)");
    EXPECT_EQ(literal("200", types::PrimitiveType::Int8), "np.int8(-56)");
    EXPECT_EQ(literal("INT32_MIN", types::PrimitiveType::Int32), "np.int32(-2147483648)");
    EXPECT_EQ(literal("true", types::PrimitiveType::Bool), "np.bool_(True)");
    EXPECT_EQ(literal("1.5f", types::PrimitiveType::Float), "np.float32(1.5)");
    EXPECT_EQ(literal("INFINITY", types::PrimitiveType::Double), "np.float64(np.inf)");
    EXPECT_EQ(literal("0x1.8p+1", types::PrimitiveType::Double), "np.float64(3)");
}

TEST(PythonLanguageExtensionTest, Expressions) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar index_type(types::PrimitiveType::Int64);
    builder.add_container("N", index_type);
    builder.add_container("i", index_type);
    codegen::PythonLanguageExtension python(builder.subject());

    auto N = symbolic::symbol("N");
    auto i = symbolic::symbol("i");
    EXPECT_EQ(python.expression(symbolic::div(N, symbolic::integer(2))), "_docc_idiv(N, 2)");
    EXPECT_EQ(python.expression(symbolic::mod(i, N)), "_docc_imod(i, N)");
    EXPECT_EQ(python.expression(symbolic::pow(N, symbolic::integer(2))), "((N) * (N))");
    EXPECT_EQ(python.expression(symbolic::__true__()), "True");
    EXPECT_EQ(python.expression(symbolic::Eq(i, N)), "(N == i)");
    EXPECT_EQ(python.expression(symbolic::Ne(i, N)), "(N != i)");

    auto both = python.expression(symbolic::And(symbolic::Lt(i, N), symbolic::Le(symbolic::integer(0), i)));
    EXPECT_NE(both.find(" and "), std::string::npos);
    EXPECT_EQ(both.find("&&"), std::string::npos);

    auto smaller = python.expression(symbolic::min(i, N));
    EXPECT_TRUE(smaller == "min(i, N)" || smaller == "min(N, i)") << smaller;
}

TEST(PythonLanguageExtensionTest, PointerSymbolsAreRejected) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("p", types::Pointer(types::Scalar(types::PrimitiveType::Float)));
    codegen::PythonLanguageExtension python(builder.subject());

    EXPECT_THROW(python.expression(symbolic::symbol("p")), std::runtime_error);
}

TEST(PythonLanguageExtensionTest, SubsetsGroupContiguousDimensions) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    codegen::PythonLanguageExtension python(builder.subject());

    types::Scalar f32(types::PrimitiveType::Float);
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    EXPECT_EQ(python.subset(types::Pointer(f32), {i}), "[i]");

    types::Array row(f32, symbolic::integer(4));
    types::Array tile(row, symbolic::integer(8));
    EXPECT_EQ(python.subset(types::Pointer(tile), {i, j, k}), "[i, j, k]");

    types::Pointer inner(f32);
    types::Pointer outer(static_cast<const types::IType&>(inner));
    EXPECT_EQ(python.subset(outer, {i, j}), "[i][j]");
    EXPECT_EQ(python.subset(f32, {}), "");
}

TEST(PythonLanguageExtensionTest, TypeCastReinterpretsBuffers) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar f32(types::PrimitiveType::Float);
    builder.add_container("A", types::Pointer(f32));
    codegen::PythonLanguageExtension python(builder.subject());

    EXPECT_EQ(python.type_cast("A", types::Pointer(f32)), "A");
    EXPECT_EQ(python.type_cast("A", types::Pointer(types::Scalar(types::PrimitiveType::Int32))), "A.view(np.int32)");
    EXPECT_EQ(
        python.type_cast("A", types::Pointer(types::Array(f32, symbolic::integer(4)))),
        "A.view(np.float32).reshape(-1, 4)"
    );
    EXPECT_EQ(python.type_cast("x", types::Scalar(types::PrimitiveType::Int64)), "np.int64(x)");
}

TEST(PythonLanguageExtensionTest, PointerOperationsAreRejected) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    codegen::PythonLanguageExtension python(builder.subject());

    EXPECT_THROW(python.address_of("a"), std::runtime_error);
    EXPECT_THROW(python.dereference("a"), std::runtime_error);
    EXPECT_THROW(python.copy_bytes("a", "b", "4"), std::runtime_error);
    EXPECT_THROW(python.reinterpret_cast_expr("f", "void (*)(int)"), std::runtime_error);
}

TEST(PythonLanguageExtensionTest, Tasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_sdiv, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    auto sdfg = builder.move();
    EXPECT_EQ(
        dispatch_python(*sdfg, sdfg->root().at(0)),
        "_in1 = np.int32(a)\n_in2 = np.int32(b)\n\n_out = _docc_sdiv(_in1, _in2)\n\nc = _out\n"
    );
}

TEST(PythonLanguageExtensionTest, ArrayTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();
    types::Scalar f32(types::PrimitiveType::Float);
    types::Pointer ptr(f32);
    builder.add_container("A", ptr, true);
    builder.add_container("B", ptr, true);

    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_neg, "_out", {"_in"});
    builder.add_computational_memlet(block, A, tasklet, "_in", {symbolic::integer(3)}, ptr);
    builder.add_computational_memlet(block, tasklet, "_out", B, {symbolic::integer(1)}, ptr);

    auto sdfg = builder.move();
    EXPECT_EQ(dispatch_python(*sdfg, sdfg->root().at(0)), "_in = np.float32((A)[3])\n\n_out = -_in\n\n(B)[1] = _out\n");
}

TEST(PythonLanguageExtensionTest, RangeForLoop) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(2))
    );

    auto sdfg = builder.move();
    EXPECT_EQ(dispatch_python(*sdfg, sdfg->root().at(0)), "for i in range(0, 10, 2):\n    pass\n");
}

TEST(PythonLanguageExtensionTest, NonAffineForLoopBecomesWhile) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    auto i = symbolic::symbol("i");
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(100)),
        symbolic::integer(1),
        symbolic::mul(i, symbolic::integer(2))
    );
    builder.add_continue(loop.root());

    auto sdfg = builder.move();
    EXPECT_EQ(dispatch_python(*sdfg, sdfg->root().at(0)), "i = 1\nwhile i < 100:\n    i = 2*i; continue\n    i = 2*i\n");
}

TEST(PythonLanguageExtensionTest, SequentialMap) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    auto i = symbolic::symbol("i");
    builder.add_map(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto sdfg = builder.move();
    EXPECT_EQ(dispatch_python(*sdfg, sdfg->root().at(0)), "# Map\nfor i in range(0, 10):\n    pass\n");
}

TEST(PythonLanguageExtensionTest, IfElseBranches) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& if_else = builder.add_if_else(builder.subject().root());
    builder.add_case(if_else, symbolic::__true__());
    builder.add_case(if_else, symbolic::__false__());

    auto sdfg = builder.move();
    EXPECT_EQ(dispatch_python(*sdfg, sdfg->root().at(0)), "if True:\n    pass\nelif False:\n    pass\n");
}

TEST(PythonLanguageExtensionTest, WhileWithBreak) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& loop = builder.add_while(builder.subject().root());
    builder.add_break(loop.root());

    auto sdfg = builder.move();
    EXPECT_EQ(dispatch_python(*sdfg, sdfg->root().at(0)), "while True:\n    break\n");
}
