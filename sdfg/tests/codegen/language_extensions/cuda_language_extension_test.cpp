#include "sdfg/codegen/language_extensions/cuda_language_extension.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"

using namespace sdfg;

TEST(CUDALanguageExtensionTest, PrimitiveType_Void) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Void);
    EXPECT_EQ(result, "void");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_Bool) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Bool);
    EXPECT_EQ(result, "bool");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_Int8) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Int8);
    EXPECT_EQ(result, "signed char");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_Int16) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Int16);
    EXPECT_EQ(result, "short");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_Int32) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Int32);
    EXPECT_EQ(result, "int");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_Int64) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Int64);
    EXPECT_EQ(result, "long long");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_UInt8) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::UInt8);
    EXPECT_EQ(result, "char");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_UInt16) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::UInt16);
    EXPECT_EQ(result, "unsigned short");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_UInt32) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::UInt32);
    EXPECT_EQ(result, "unsigned int");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_UInt64) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::UInt64);
    EXPECT_EQ(result, "unsigned long long");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_Float) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Float);
    EXPECT_EQ(result, "float");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_Double) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.primitive_type(types::PrimitiveType::Double);
    EXPECT_EQ(result, "double");
}

TEST(CUDALanguageExtensionTest, Declaration_Scalar) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.declaration("var", types::Scalar(types::PrimitiveType::Int32));
    EXPECT_EQ(result, "int var");
}

TEST(CUDALanguageExtensionTest, Declaration_Pointer) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.declaration("var", types::Pointer(types::Scalar(types::PrimitiveType::Int32)));
    EXPECT_EQ(result, "int *var");

    result = generator.declaration("var", types::Pointer());
    EXPECT_EQ(result, "void* var");
}

TEST(CUDALanguageExtensionTest, Declaration_Array) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result =
        generator.declaration("var", types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)));
    EXPECT_EQ(result, "int var[10]");
}

TEST(CUDALanguageExtensionTest, Declaration_Struct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.declaration("var", types::Structure("MyStruct"));
    EXPECT_EQ(result, "MyStruct var");
}

TEST(CUDALanguageExtensionTest, Declaration_ArrayOfStruct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.declaration("var", types::Array(types::Structure("MyStruct"), symbolic::integer(10)));
    EXPECT_EQ(result, "MyStruct var[10]");
}

TEST(CUDALanguageExtensionTest, Declaration_PointerToArray) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.declaration(
        "var", types::Pointer(types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)))
    );
    EXPECT_EQ(result, "int (*var)[10]");
}

TEST(CUDALanguageExtensionTest, Typecast) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.type_cast("var", types::Pointer(types::Scalar(types::PrimitiveType::Float)));
    EXPECT_EQ(result, "reinterpret_cast<float *>(var)");
}

TEST(CUDALanguageExtensionTest, SubsetToCpp_Scalar) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.subset(types::Scalar(types::PrimitiveType::Int32), data_flow::Subset());
    EXPECT_EQ(result, "");
}

TEST(CUDALanguageExtensionTest, SubsetToCpp_Array) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.subset(
        types::Array(types::Scalar(types::PrimitiveType::Int32), symbolic::integer(10)),
        data_flow::Subset{symbolic::integer(1)}
    );
    EXPECT_EQ(result, "[1]");
}

TEST(CUDALanguageExtensionTest, SubsetToCpp_Struct) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& struct_def = builder.add_structure("MyStruct", false);
    struct_def.add_member(types::Scalar(types::PrimitiveType::Int32));
    struct_def.add_member(types::Scalar(types::PrimitiveType::Float));

    codegen::CUDALanguageExtension generator(sdfg);
    auto result = generator.subset(types::Structure("MyStruct"), data_flow::Subset{symbolic::integer(1)});
    EXPECT_EQ(result, ".member_1");
}

TEST(CUDALanguageExtensionTest, PrimitiveType_Complex) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CUDALanguageExtension generator(sdfg);

    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CHalf), "__daisy_type_complex_half");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CBFloat), "__daisy_type_complex_bfloat");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CFloat), "__daisy_type_complex_float");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CDouble), "__daisy_type_complex_double");
    EXPECT_EQ(generator.primitive_type(types::PrimitiveType::CFP128), "__daisy_type_complex_fp128");
}

TEST(CUDALanguageExtensionTest, Zero_Complex) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CUDALanguageExtension generator(sdfg);

    EXPECT_EQ(generator.zero(types::PrimitiveType::CFloat), "__daisy_type_complex_float{0, 0}");
    EXPECT_EQ(generator.zero(types::PrimitiveType::CDouble), "__daisy_type_complex_double{0, 0}");
}
