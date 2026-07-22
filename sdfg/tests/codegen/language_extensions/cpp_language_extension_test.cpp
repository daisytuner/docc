#include "sdfg/codegen/language_extensions/cpp_language_extension.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/types/utils.h"

using namespace sdfg;

TEST(CPPLanguageExtensionTest, PrimitiveType_Void) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Void);
    EXPECT_EQ(result, "void");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Bool) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Bool);
    EXPECT_EQ(result, "bool");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Int8) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Int8);
    EXPECT_EQ(result, "int8_t");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Int16) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Int16);
    EXPECT_EQ(result, "int16_t");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Int32) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Int32);
    EXPECT_EQ(result, "int32_t");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Int64) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Int64);
    EXPECT_EQ(result, "int64_t");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_UInt8) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::UInt8);
    EXPECT_EQ(result, "uint8_t");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_UInt16) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::UInt16);
    EXPECT_EQ(result, "uint16_t");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_UInt32) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::UInt32);
    EXPECT_EQ(result, "uint32_t");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_UInt64) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::UInt64);
    EXPECT_EQ(result, "uint64_t");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Float) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Float);
    EXPECT_EQ(result, "float");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Double) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Double);
    EXPECT_EQ(result, "double");
}

TEST(CPPLanguageExtensionTest, Declaration_Scalar) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.declaration("var", types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(result, "int32_t var");
}

TEST(CPPLanguageExtensionTest, Declaration_Pointer) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.declaration("var", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));
    EXPECT_EQ(result, "int32_t *var");
    result = generator.declaration("var", types::Pointer());
    EXPECT_EQ(result, "void* var");
}

TEST(CPPLanguageExtensionTest, Declaration_Array) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result =
        generator.declaration("var", types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)));
    EXPECT_EQ(result, "int32_t var[10]");
}

TEST(CPPLanguageExtensionTest, Declaration_Struct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.declaration("var", types::Structure("MyStruct"));
    EXPECT_EQ(result, "MyStruct var");
}

TEST(CPPLanguageExtensionTest, Declaration_ArrayOfStruct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.declaration("var", types::Array(types::Structure("MyStruct"), symbolic::integer(10)));
    EXPECT_EQ(result, "MyStruct var[10]");
}

TEST(CPPLanguageExtensionTest, Declaration_PointerToArray) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.declaration(
        "var", types::Pointer(types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)))
    );
    EXPECT_EQ(result, "int32_t (*var)[10]");
}

TEST(CPPLanguageExtensionTest, Typecast) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.type_cast("var", types::Pointer(types::Scalar(types::PrimitiveType::Float)));
    EXPECT_EQ(result, "reinterpret_cast<float *>(var)");
}

TEST(CPPLanguageExtensionTest, Sizeof) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto type = types::Pointer(types::Structure("some_t"));
    auto size_expr = types::get_contiguous_element_size(type, true);
    auto result = generator.expression(size_expr);
    EXPECT_EQ(result, "sizeof(some_t )");
}

TEST(CPPLanguageExtensionTest, SubsetToCpp_Scalar) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.subset(types::Scalar(types::PrimitiveType::Int32), data_flow::Subset());
    EXPECT_EQ(result, "");
}

TEST(CPPLanguageExtensionTest, SubsetToCpp_Array) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.subset(
        types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)),
        data_flow::Subset{symbolic::integer(1)}
    );
    EXPECT_EQ(result, "[1]");
}

TEST(CPPLanguageExtensionTest, SubsetToCpp_Struct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& struct_def = builder.add_structure("MyStruct", false);
    struct_def.add_member(types::Scalar(types::PrimitiveType::Int32));
    struct_def.add_member(types::Scalar(types::PrimitiveType::Float));

    codegen::CPPLanguageExtension generator(sdfg);
    auto result = generator.subset(types::Structure("MyStruct"), data_flow::Subset{symbolic::integer(1)});
    EXPECT_EQ(result, ".member_1");
}

TEST(CPPLanguageExtensionTest, Expression_Pow2) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);

    auto sym = symbolic::symbol("x");
    auto result = generator.expression(symbolic::pow(sym, symbolic::integer(2)));
    EXPECT_EQ(result, "((x) * (x))");
}

TEST(CPPLanguageExtensionTest, Expression_Pow2_Mul) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CPPLanguageExtension generator(sdfg);

    auto sym = symbolic::symbol("x");
    auto result = generator.expression(symbolic::mul(sym, sym));
    EXPECT_EQ(result, "((x) * (x))");
}

TEST(CPPLanguageExtensionTest, Expression_External) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    builder.add_container("EXT1", types::Scalar(types::PrimitiveType::Int32), false, true);

    codegen::CPPLanguageExtension generator(sdfg);

    auto sym = symbolic::symbol("EXT1");
    auto result = generator.expression(sym);
    EXPECT_EQ(result, "(reinterpret_cast<uintptr_t>(&EXT1))");
}

TEST(CPPLanguageExtensionTest, PrimitiveType_Complex) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CPPLanguageExtension generator(sdfg);

    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CHalf), "half2");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CBFloat), "bfloat162");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CFloat), "float2");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CDouble), "double2");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CFP128), "fp128_2");
}

TEST(CPPLanguageExtensionTest, Zero_Complex) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CPPLanguageExtension generator(sdfg);

    EXPECT_EQ(generator.zero(types::PrimitiveType::CFloat), "float2{0, 0}");
    EXPECT_EQ(generator.zero(types::PrimitiveType::CDouble), "double2{0, 0}");
    EXPECT_EQ(generator.zero(types::PrimitiveType::CFP128), "fp128_2{0, 0}");
}

TEST(CPPLanguageExtensionTest, Tasklet_ComplexAdd) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& state = builder.add_state();
    types::Scalar desc(types::PrimitiveType::CFloat);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    auto& a = builder.add_access(state, "a");
    auto& b = builder.add_access(state, "b");
    auto& c = builder.add_access(state, "c");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::complex_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, a, tasklet, "_in1", {});
    builder.add_computational_memlet(state, b, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", c, {});
    codegen::CPPLanguageExtension generator(builder.subject());
    EXPECT_EQ(generator.tasklet(tasklet), "__daisy_cadd_f(_in1, _in2)");
}

TEST(CPPLanguageExtensionTest, Tasklet_ComplexDiv) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& state = builder.add_state();
    types::Scalar desc(types::PrimitiveType::CDouble);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);
    auto& a = builder.add_access(state, "a");
    auto& b = builder.add_access(state, "b");
    auto& c = builder.add_access(state, "c");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::complex_div, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, a, tasklet, "_in1", {});
    builder.add_computational_memlet(state, b, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", c, {});
    codegen::CPPLanguageExtension generator(builder.subject());
    EXPECT_EQ(generator.tasklet(tasklet), "__daisy_cdiv_d(_in1, _in2)");
}
