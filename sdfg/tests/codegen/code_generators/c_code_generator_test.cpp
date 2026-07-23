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
    EXPECT_EQ(result, "    long long t0;\n");
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
    EXPECT_EQ(result, "    arg0 = malloc(8);\n    free(arg0);\n");
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
    EXPECT_EQ(result, "    long long *t0;\n    t0 = malloc(8);\n    free(t0);\n");
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
    EXPECT_EQ(result, "    arg0 = malloc(8);\n");
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
    EXPECT_EQ(result, "    long long *t0;\n    t0 = malloc(8);\n");
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
    EXPECT_EQ(result, "    free(arg0);\n");
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
