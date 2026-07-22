#include "sdfg/codegen/code_generators/c_code_generator.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(CCodeGeneratorTest, FunctionDefintion) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*sdfg);

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::outermost_loops_plan(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    auto result = generator.function_definition();
    EXPECT_EQ(result, "extern void sdfg_a(void)");
}

TEST(CCodeGeneratorTest, Allocation_Stack_Argument) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    builder.add_container("arg0", types::Scalar(types::PrimitiveType::Int64), true, false);

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    analysis::AnalysisManager analysis_manager(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    auto result = generator.main().str();
    EXPECT_EQ(result, "");
}

TEST(CCodeGeneratorTest, Allocation_Stack_Transient) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    builder.add_container("t0", types::Scalar(types::PrimitiveType::Int64), false, false);

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    analysis::AnalysisManager analysis_manager(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    auto result = generator.main().str();
    EXPECT_EQ(result, "long long t0;\n");
}

TEST(CCodeGeneratorTest, Allocation_Heap_Argument_Managed) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    types::Scalar long_type(types::PrimitiveType::Int64);
    types::Pointer pointer_type(
        types::StorageType::
            CPU_Heap(symbolic::integer(8), types::StorageType::AllocationType::Managed, types::StorageType::Managed),
        0,
        "",
        long_type
    );
    builder.add_container("arg0", pointer_type, true, false);

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    analysis::AnalysisManager analysis_manager(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    auto result = generator.main().str();
    EXPECT_EQ(result, "arg0 = malloc(8);\nfree(arg0);\n");
}

TEST(CCodeGeneratorTest, Allocation_Heap_Transient_Managed) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    types::Scalar long_type(types::PrimitiveType::Int64);
    types::Pointer pointer_type(
        types::StorageType::
            CPU_Heap(symbolic::integer(8), types::StorageType::AllocationType::Managed, types::StorageType::Managed),
        0,
        "",
        long_type
    );
    builder.add_container("t0", pointer_type, false, false);

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    analysis::AnalysisManager analysis_manager(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    auto result = generator.main().str();
    EXPECT_EQ(result, "long long *t0;\nt0 = malloc(8);\nfree(t0);\n");
}

TEST(CCodeGeneratorTest, Allocation_Heap_Argument_Default_Lifetime) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    types::Scalar long_type(types::PrimitiveType::Int64);
    types::Pointer pointer_type(
        types::StorageType::
            CPU_Heap(symbolic::integer(8), types::StorageType::AllocationType::Managed, types::StorageType::Unmanaged),
        0,
        "",
        long_type
    );
    builder.add_container("arg0", pointer_type, true, false);

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    analysis::AnalysisManager analysis_manager(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    auto result = generator.main().str();
    EXPECT_EQ(result, "arg0 = malloc(8);\n");
}

TEST(CCodeGeneratorTest, Allocation_Heap_Transient_Default_Lifetime) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    types::Scalar long_type(types::PrimitiveType::Int64);
    types::Pointer pointer_type(
        types::StorageType::
            CPU_Heap(symbolic::integer(8), types::StorageType::AllocationType::Managed, types::StorageType::Unmanaged),
        0,
        "",
        long_type
    );
    builder.add_container("t0", pointer_type, false, false);

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    analysis::AnalysisManager analysis_manager(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    auto result = generator.main().str();
    EXPECT_EQ(result, "long long *t0;\nt0 = malloc(8);\n");
}

TEST(CCodeGeneratorTest, Deallocation_Heap_Argument_SDFG_Lifetime) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    types::Scalar long_type(types::PrimitiveType::Int64);
    types::Pointer pointer_type(
        types::StorageType::
            CPU_Heap(SymEngine::null, types::StorageType::AllocationType::Unmanaged, types::StorageType::Managed),
        0,
        "",
        long_type
    );
    builder.add_container("arg0", pointer_type, true, false);

    auto sdfg = builder.move();

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    analysis::AnalysisManager analysis_manager(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    generator.generate();
    auto result = generator.main().str();
    EXPECT_EQ(result, "free(arg0);\n");
}

TEST(CCodeGeneratorTest, DispatchStructures_Basic) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*sdfg);

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    EXPECT_TRUE(generator.generate());

    auto result = generator.classes().str();
    EXPECT_EQ(result, R"(typedef struct MyStructA MyStructA;
typedef struct MyStructA
{
char member_0;
} MyStructA;
)");
}

TEST(CCodeGeneratorTest, DispatchStructures_Nested) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    auto& struct_def_A = builder.add_structure("MyStructA", false);
    struct_def_A.add_member(types::Scalar(types::PrimitiveType::UInt8));

    auto& struct_def_B = builder.add_structure("MyStructB", false);
    struct_def_B.add_member(types::Structure("MyStructA"));

    auto sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*sdfg);

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    EXPECT_TRUE(generator.generate());

    auto result = generator.classes().str();
    EXPECT_EQ(result, R"(typedef struct MyStructB MyStructB;
typedef struct MyStructA MyStructA;
typedef struct MyStructA
{
char member_0;
} MyStructA;
typedef struct MyStructB
{
MyStructA member_0;
} MyStructB;
)");
}

TEST(CCodeGeneratorTest, DispatchGlobals) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Pointer ptr_type(base_type);
    builder.add_container("a", ptr_type, false, true);

    auto sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*sdfg);

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    EXPECT_TRUE(generator.generate());

    auto result = generator.globals().str();
    EXPECT_EQ(result, "extern int a;\n");
}

TEST(CCodeGeneratorTest, ComplexSupportPreamble) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    auto sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*sdfg);

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    EXPECT_TRUE(generator.generate());

    auto includes = generator.includes().str();
    // Reused vector types are defined on CPU targets.
    EXPECT_NE(includes.find("typedef struct { float x; float y; } float2;"), std::string::npos);
    EXPECT_NE(includes.find("typedef struct { double x; double y; } double2;"), std::string::npos);
    EXPECT_NE(includes.find("typedef struct { _Float16 x; _Float16 y; } half2;"), std::string::npos);
    EXPECT_NE(includes.find("typedef struct { __bf16 x; __bf16 y; } bfloat162;"), std::string::npos);
    EXPECT_NE(includes.find("fp128_2;"), std::string::npos);
    // Helper functions are instantiated for each element suffix.
    EXPECT_NE(includes.find("__daisy_cadd_##SUF"), std::string::npos);
    EXPECT_NE(includes.find("__DAISY_DEFINE_COMPLEX(float2, float, float, f,"), std::string::npos);
    EXPECT_NE(includes.find("__DAISY_DEFINE_COMPLEX(double2, double, double, d,"), std::string::npos);
}

TEST(CCodeGeneratorTest, ComplexAddSchedule) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::CFloat);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", desc, true);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::complex_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, a, tasklet, "_in1", {});
    builder.add_computational_memlet(block, b, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", c, {});

    auto sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*sdfg);

    auto instrumentation_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(*sdfg);
    codegen::CCodeGenerator generator(*sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan);
    EXPECT_TRUE(generator.generate());

    auto result = generator.main().str();
    EXPECT_NE(result.find("__daisy_cadd_f(_in1, _in2)"), std::string::npos);
    EXPECT_NE(result.find("float2 _in1"), std::string::npos);
}
