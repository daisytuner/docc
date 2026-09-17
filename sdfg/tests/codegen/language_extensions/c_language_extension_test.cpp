#include "sdfg/codegen/language_extensions/c_language_extension.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"

#include "sdfg/types/structure.h"
#include "sdfg/types/utils.h"

using namespace sdfg;

TEST(CLanguageExtensionTest, PrimitiveType_Void) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::Void);
    EXPECT_EQ(result, "void");
}

TEST(CLanguageExtensionTest, PrimitiveType_Bool) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::Bool);
    EXPECT_EQ(result, "bool");
}

TEST(CLanguageExtensionTest, PrimitiveType_Int8) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::Int8);
    EXPECT_EQ(result, "signed char");
}

TEST(CLanguageExtensionTest, PrimitiveType_Int16) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::Int16);
    EXPECT_EQ(result, "short");
}

TEST(CLanguageExtensionTest, PrimitiveType_Int32) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::Int32);
    EXPECT_EQ(result, "int");
}

TEST(CLanguageExtensionTest, PrimitiveType_Int64) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::Int64);
    EXPECT_EQ(result, "long long");
}

TEST(CLanguageExtensionTest, PrimitiveType_UInt8) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::UInt8);
    EXPECT_EQ(result, "char");
}

TEST(CLanguageExtensionTest, PrimitiveType_UInt16) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::UInt16);
    EXPECT_EQ(result, "unsigned short");
}

TEST(CLanguageExtensionTest, PrimitiveType_UInt32) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::UInt32);
    EXPECT_EQ(result, "unsigned int");
}

TEST(CLanguageExtensionTest, PrimitiveType_UInt64) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::UInt64);
    EXPECT_EQ(result, "unsigned long long");
}

TEST(CLanguageExtensionTest, PrimitiveType_Float) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::Float);
    EXPECT_EQ(result, "float");
}

TEST(CLanguageExtensionTest, PrimitiveType_Double) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.primitive_type(types::PrimitiveType::Double);
    EXPECT_EQ(result, "double");
}

TEST(CLanguageExtensionTest, Declaration_Scalar) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.declaration("var", types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(result, "int var");
}

TEST(CLanguageExtensionTest, Declaration_Pointer) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.declaration("var", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));
    EXPECT_EQ(result, "int *var");

    result = generator.declaration("var", types::Pointer());
    EXPECT_EQ(result, "void* var");
}

TEST(CLanguageExtensionTest, Declaration_Array) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result =
        generator.declaration("var", types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)));
    EXPECT_EQ(result, "int var[10]");
}

TEST(CLanguageExtensionTest, Declaration_Struct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.declaration("var", types::Structure("MyStruct"));
    EXPECT_EQ(result, "MyStruct var");
}

TEST(CLanguageExtensionTest, Declaration_ArrayOfStruct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.declaration("var", types::Array(types::Structure("MyStruct"), symbolic::integer(10)));
    EXPECT_EQ(result, "MyStruct var[10]");
}

TEST(CLanguageExtensionTest, Declaration_PointerToArray) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.declaration(
        "var", types::Pointer(types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)))
    );
    EXPECT_EQ(result, "int (*var)[10]");
}

TEST(CLanguageExtensionTest, Typecast) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.type_cast("var", types::Pointer(types::Scalar(types::PrimitiveType::Float)));
    EXPECT_EQ(result, "(float *) var");
}

TEST(CLanguageExtensionTest, Sizeof) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto type = types::Pointer(types::Structure("some_t"));
    auto size_expr = types::get_contiguous_element_size(type, true);
    auto result = generator.expression(size_expr);
    EXPECT_EQ(result, "sizeof(some_t )");
}

TEST(CLanguageExtensionTest, SubsetToCpp_Scalar) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.subset(types::Scalar(types::PrimitiveType::Int32), data_flow::Subset());
    EXPECT_EQ(result, "");
}

TEST(CLanguageExtensionTest, SubsetToCpp_Array) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.subset(
        types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)),
        data_flow::Subset{symbolic::integer(1)}
    );
    EXPECT_EQ(result, "[1]");
}

TEST(CLanguageExtensionTest, SubsetToCpp_Struct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& struct_def = builder.add_structure("MyStruct", false);
    struct_def.add_member(types::Scalar(types::PrimitiveType::Int32));
    struct_def.add_member(types::Scalar(types::PrimitiveType::Float));

    codegen::CLanguageExtension generator(sdfg);

    auto result = generator.subset(types::Structure("MyStruct"), data_flow::Subset{symbolic::integer(1)});
    EXPECT_EQ(result, ".member_1");
}

TEST(CLanguageExtensionTest, Expression_Pow2) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto sym = symbolic::symbol("x");
    auto result = generator.expression(symbolic::pow(sym, symbolic::integer(2)));
    EXPECT_EQ(result, "((x) * (x))");
}

TEST(CLanguageExtensionTest, Expression_Pow2_Mul) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    auto sym = symbolic::symbol("x");
    auto result = generator.expression(symbolic::mul(sym, sym));
    EXPECT_EQ(result, "((x) * (x))");
}

TEST(CLanguageExtensionTest, Expression_External) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    builder.add_container("EXT1", types::Scalar(types::PrimitiveType::Int32), false, true);

    codegen::CLanguageExtension generator(sdfg);

    auto sym = symbolic::symbol("EXT1");
    auto result = generator.expression(sym);
    EXPECT_EQ(result, "((uintptr_t) (&EXT1))");
}

TEST(CLanguageExtensionTest, PrimitiveType_Complex) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CHalf), "__daisy_type_complex_half");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CBFloat), "__daisy_type_complex_bfloat");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CFloat), "__daisy_type_complex_float");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CDouble), "__daisy_type_complex_double");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CFP128), "__daisy_type_complex_fp128");
}

TEST(CLanguageExtensionTest, Zero_Complex) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension generator(sdfg);

    EXPECT_EQ(generator.zero(types::PrimitiveType::CFloat), "(__daisy_type_complex_float){0, 0}");
    EXPECT_EQ(generator.zero(types::PrimitiveType::CDouble), "(__daisy_type_complex_double){0, 0}");
    EXPECT_EQ(generator.zero(types::PrimitiveType::CHalf), "(__daisy_type_complex_half){0, 0}");
    EXPECT_EQ(generator.zero(types::PrimitiveType::CBFloat), "(__daisy_type_complex_bfloat){0, 0}");
    EXPECT_EQ(generator.zero(types::PrimitiveType::CFP128), "(__daisy_type_complex_fp128){0, 0}");
}
