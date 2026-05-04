#include <gtest/gtest.h>
#include <memory>
#include <sdfg/serializer/json_serializer.h>
#include <symengine/eval.h>
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda_map_dispatcher.h"
#include "sdfg/transformations/offloading/cuda_transform.h"

#include "sdfg/targets/cuda/cuda.h"

namespace sdfg::cuda {

TEST(CUDAKernel, DispatcherTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    auto& indvar = builder.add_container("i", int_desc);
    auto& A_device = builder.add_container("__daisy_cuda_A", pointer_type);
    auto& B_host = builder.add_container("B", base_desc);

    // Create a map with CUDA schedule
    ScheduleType cuda_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule, CUDADimension::X);
    ScheduleType_CUDA::block_size(cuda_schedule, symbolic::integer(32));

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& map = builder.add_map(root, symbolic::symbol("i"), condition, init, update, cuda_schedule);

    auto& block = builder.add_block(map.root());
    auto& access = builder.add_access(block, "__daisy_cuda_A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& constant = builder.add_constant(block, "0.0f", types::Scalar(types::PrimitiveType::Float));

    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, types::Scalar(types::PrimitiveType::Float));
    builder.add_computational_memlet(block, tasklet, "out_", access, {symbolic::symbol("i")}, pointer_type);

    auto& block2 = builder.add_block(root);
    auto& access2 = builder.add_access(block2, "__daisy_cuda_A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& access_B = builder.add_access(block2, "B");

    builder.add_computational_memlet(block2, access2, tasklet2, "in_", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block2, tasklet2, "out_", access_B, {}, base_desc);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAMapDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, map, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);
    std::string kernel_name = "kernel_test_sdfg_1";

    // Check if the generated code contains the expected function call
    EXPECT_EQ(library_snippet_factory.snippets().size(), 1);

    EXPECT_TRUE(library_snippet_factory.globals_snippets().count("#include <cstdio>"));
    EXPECT_EQ(globals_stream.str(), "__global__ void " + kernel_name + "(float *__daisy_cuda_A);\n");

    EXPECT_EQ(
        main_stream.str(),
        "{\n    " + kernel_name +
            "<<<dim3((int)(4), (int)(1), (int)(1)), dim3((int)(32), (int)(1), (int)(1))>>>(__daisy_cuda_A);\n}\n"
    );
    EXPECT_EQ(
        library_snippet_factory.snippets().at(kernel_name).stream().str(),
        "#include \"\"\n\n__global__ void kernel_test_sdfg_1(float *__daisy_cuda_A){\n    int "
        "__daisy_cuda_thread_idx_x = threadIdx.x;\n    int __daisy_cuda_indvar_x = threadIdx.x + "
        "blockIdx.x*blockDim.x;\n    int __daisy_cuda_thread_idx_y = threadIdx.y;\n    int __daisy_cuda_indvar_y = "
        "threadIdx.y + blockIdx.y*blockDim.y;\n    int __daisy_cuda_thread_idx_z = threadIdx.z;\n    int "
        "__daisy_cuda_indvar_z = threadIdx.z + blockIdx.z*blockDim.z;\n    int i = __daisy_cuda_indvar_x;\n    if (i < "
        "100) {\n            {\n                float in_ = 0.0f;\n                float out_;\n\n                out_ "
        "= in_;\n\n                (reinterpret_cast<float *>(__daisy_cuda_A))[i] = out_;\n            }\n    }\n}\n"
    );
}

TEST(CUDAKernel, NestedXYDispatcherTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    auto& indvar = builder.add_container("i", int_desc);
    auto& jndvar = builder.add_container("j", int_desc);
    auto& A_device = builder.add_container("__daisy_cuda_A", pointer_type);
    auto& B_host = builder.add_container("B", base_desc);

    // Create a map with CUDA schedule
    ScheduleType cuda_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule, CUDADimension::X);
    ScheduleType_CUDA::block_size(cuda_schedule, symbolic::integer(32));

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& map = builder.add_map(root, symbolic::symbol("i"), condition, init, update, cuda_schedule);

    ScheduleType cuda_schedule2 = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule2, CUDADimension::Y);
    ScheduleType_CUDA::block_size(cuda_schedule2, symbolic::integer(4));

    auto condition2 = symbolic::Lt(symbolic::symbol("j"), symbolic::integer(40));
    auto init2 = symbolic::integer(0);
    auto update2 = symbolic::add(symbolic::symbol("j"), symbolic::integer(1));

    auto& map2 = builder.add_map(map.root(), symbolic::symbol("j"), condition2, init2, update2, cuda_schedule2);

    auto& block = builder.add_block(map2.root());
    auto& access = builder.add_access(block, "__daisy_cuda_A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& constant = builder.add_constant(block, "0.0f", types::Scalar(types::PrimitiveType::Float));

    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, types::Scalar(types::PrimitiveType::Float));
    builder.add_computational_memlet(
        block, tasklet, "out_", access, {symbolic::add(symbolic::symbol("i"), symbolic::symbol("j"))}, pointer_type
    );

    auto& block2 = builder.add_block(root);
    auto& access2 = builder.add_access(block2, "__daisy_cuda_A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& access_B = builder.add_access(block2, "B");

    builder.add_computational_memlet(block2, access2, tasklet2, "in_", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block2, tasklet2, "out_", access_B, {}, base_desc);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAMapDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, map, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);
    std::string kernel_name = "kernel_test_sdfg_1";

    // Check if the generated code contains the expected function call
    EXPECT_EQ(library_snippet_factory.snippets().size(), 1);

    EXPECT_TRUE(library_snippet_factory.globals_snippets().count("#include <cstdio>"));
    EXPECT_EQ(globals_stream.str(), "__global__ void " + kernel_name + "(float *__daisy_cuda_A);\n");

    EXPECT_EQ(
        main_stream.str(),
        "{\n    " + kernel_name +
            "<<<dim3((int)(4), (int)(10), (int)(1)), dim3((int)(32), (int)(4), (int)(1))>>>(__daisy_cuda_A);\n}\n"
    );
    EXPECT_EQ(
        library_snippet_factory.snippets().at(kernel_name).stream().str(),
        "#include \"\"\n\n__global__ void kernel_test_sdfg_1(float *__daisy_cuda_A){\n    int "
        "__daisy_cuda_thread_idx_x = threadIdx.x;\n    int __daisy_cuda_indvar_x = threadIdx.x + "
        "blockIdx.x*blockDim.x;\n    int __daisy_cuda_thread_idx_y = threadIdx.y;\n    int __daisy_cuda_indvar_y = "
        "threadIdx.y + blockIdx.y*blockDim.y;\n    int __daisy_cuda_thread_idx_z = threadIdx.z;\n    int "
        "__daisy_cuda_indvar_z = threadIdx.z + blockIdx.z*blockDim.z;\n    int i = __daisy_cuda_indvar_x;\n    int j = "
        "__daisy_cuda_indvar_y;\n    if (i < 100) {\n            if (j < 40) {\n                    {\n                "
        "        float in_ = 0.0f;\n                        float out_;\n\n                        out_ = in_;\n\n     "
        "                   (reinterpret_cast<float *>(__daisy_cuda_A))[i + j] = out_;\n                    }\n        "
        "    }\n    }\n}\n"
    );
}

TEST(CUDAKernel, NestedXZDispatcherTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    auto& indvar = builder.add_container("i", int_desc);
    auto& jndvar = builder.add_container("j", int_desc);
    auto& A_device = builder.add_container("__daisy_cuda_A", pointer_type);
    auto& B_host = builder.add_container("B", base_desc);

    // Create a map with CUDA schedule
    ScheduleType cuda_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule, CUDADimension::X);
    ScheduleType_CUDA::block_size(cuda_schedule, symbolic::integer(32));

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& map = builder.add_map(root, symbolic::symbol("i"), condition, init, update, cuda_schedule);

    ScheduleType cuda_schedule2 = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule2, CUDADimension::Z);
    ScheduleType_CUDA::block_size(cuda_schedule2, symbolic::integer(4));

    auto condition2 = symbolic::Lt(symbolic::symbol("j"), symbolic::integer(40));
    auto init2 = symbolic::integer(0);
    auto update2 = symbolic::add(symbolic::symbol("j"), symbolic::integer(1));

    auto& map2 = builder.add_map(map.root(), symbolic::symbol("j"), condition2, init2, update2, cuda_schedule2);

    auto& block = builder.add_block(map2.root());
    auto& access = builder.add_access(block, "__daisy_cuda_A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& constant = builder.add_constant(block, "0.0f", types::Scalar(types::PrimitiveType::Float));

    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, types::Scalar(types::PrimitiveType::Float));
    builder.add_computational_memlet(
        block, tasklet, "out_", access, {symbolic::add(symbolic::symbol("i"), symbolic::symbol("j"))}, pointer_type
    );

    auto& block2 = builder.add_block(root);
    auto& access2 = builder.add_access(block2, "__daisy_cuda_A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& access_B = builder.add_access(block2, "B");

    builder.add_computational_memlet(block2, access2, tasklet2, "in_", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block2, tasklet2, "out_", access_B, {}, base_desc);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAMapDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, map, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);
    std::string kernel_name = "kernel_test_sdfg_1";

    // Check if the generated code contains the expected function call
    EXPECT_EQ(library_snippet_factory.snippets().size(), 1);

    EXPECT_TRUE(library_snippet_factory.globals_snippets().count("#include <cstdio>"));
    EXPECT_EQ(globals_stream.str(), "__global__ void " + kernel_name + "(float *__daisy_cuda_A);\n");

    EXPECT_EQ(
        main_stream.str(),
        "{\n    " + kernel_name +
            "<<<dim3((int)(4), (int)(1), (int)(10)), dim3((int)(32), (int)(1), (int)(4))>>>(__daisy_cuda_A);\n}\n"
    );
    EXPECT_EQ(
        library_snippet_factory.snippets().at(kernel_name).stream().str(),
        "#include \"\"\n\n__global__ void kernel_test_sdfg_1(float *__daisy_cuda_A){\n    int "
        "__daisy_cuda_thread_idx_x = threadIdx.x;\n    int __daisy_cuda_indvar_x = threadIdx.x + "
        "blockIdx.x*blockDim.x;\n    int __daisy_cuda_thread_idx_y = threadIdx.y;\n    int __daisy_cuda_indvar_y = "
        "threadIdx.y + blockIdx.y*blockDim.y;\n    int __daisy_cuda_thread_idx_z = threadIdx.z;\n    int "
        "__daisy_cuda_indvar_z = threadIdx.z + blockIdx.z*blockDim.z;\n    int i = __daisy_cuda_indvar_x;\n    int j = "
        "__daisy_cuda_indvar_z;\n    if (i < 100) {\n            if (j < 40) {\n                    {\n                "
        "        float in_ = 0.0f;\n                        float out_;\n\n                        out_ = in_;\n\n     "
        "                   (reinterpret_cast<float *>(__daisy_cuda_A))[i + j] = out_;\n                    }\n        "
        "    }\n    }\n}\n"
    );
}

TEST(CUDAKernel, NestedXYZDispatcherTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    auto& indvar = builder.add_container("i", int_desc);
    auto& jndvar = builder.add_container("j", int_desc);
    auto& kndvar = builder.add_container("k", int_desc);
    auto& A_device = builder.add_container("__daisy_cuda_A", pointer_type);
    auto& B_host = builder.add_container("B", base_desc);

    // Create a map with CUDA schedule
    ScheduleType cuda_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule, CUDADimension::X);
    ScheduleType_CUDA::block_size(cuda_schedule, symbolic::integer(32));

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& map = builder.add_map(root, symbolic::symbol("i"), condition, init, update, cuda_schedule);

    ScheduleType cuda_schedule2 = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule2, CUDADimension::Y);
    ScheduleType_CUDA::block_size(cuda_schedule2, symbolic::integer(4));

    auto condition2 = symbolic::Lt(symbolic::symbol("j"), symbolic::integer(40));
    auto init2 = symbolic::integer(0);
    auto update2 = symbolic::add(symbolic::symbol("j"), symbolic::integer(1));

    auto& map2 = builder.add_map(map.root(), symbolic::symbol("j"), condition2, init2, update2, cuda_schedule2);

    ScheduleType cuda_schedule3 = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule3, CUDADimension::Z);
    ScheduleType_CUDA::block_size(cuda_schedule3, symbolic::integer(2));

    auto condition3 = symbolic::Lt(symbolic::symbol("k"), symbolic::integer(200));
    auto init3 = symbolic::integer(0);
    auto update3 = symbolic::add(symbolic::symbol("k"), symbolic::integer(1));

    auto& map3 = builder.add_map(map2.root(), symbolic::symbol("k"), condition3, init3, update3, cuda_schedule3);


    auto& block = builder.add_block(map3.root());
    auto& access = builder.add_access(block, "__daisy_cuda_A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& constant = builder.add_constant(block, "0.0f", types::Scalar(types::PrimitiveType::Float));

    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, types::Scalar(types::PrimitiveType::Float));
    builder.add_computational_memlet(
        block,
        tasklet,
        "out_",
        access,
        {symbolic::add(symbolic::add(symbolic::symbol("i"), symbolic::symbol("j")), symbolic::symbol("k"))},
        pointer_type
    );

    auto& block2 = builder.add_block(root);
    auto& access2 = builder.add_access(block2, "__daisy_cuda_A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& access_B = builder.add_access(block2, "B");

    builder.add_computational_memlet(block2, access2, tasklet2, "in_", {symbolic::zero()}, pointer_type);
    builder.add_computational_memlet(block2, tasklet2, "out_", access_B, {}, base_desc);

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());
    CUDAMapDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, map, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);
    std::string kernel_name = "kernel_test_sdfg_1";

    // Check if the generated code contains the expected function call
    EXPECT_EQ(library_snippet_factory.snippets().size(), 1);

    EXPECT_TRUE(library_snippet_factory.globals_snippets().count("#include <cstdio>"));
    EXPECT_EQ(globals_stream.str(), "__global__ void " + kernel_name + "(float *__daisy_cuda_A);\n");

    EXPECT_EQ(
        main_stream.str(),
        "{\n    kernel_test_sdfg_1<<<dim3((int)(4), (int)(10), (int)(100)), dim3((int)(32), (int)(4), "
        "(int)(2))>>>(__daisy_cuda_A);\n}\n"
    );
    EXPECT_EQ(
        library_snippet_factory.snippets().at(kernel_name).stream().str(),
        "#include \"\"\n\n__global__ void kernel_test_sdfg_1(float *__daisy_cuda_A){\n    int "
        "__daisy_cuda_thread_idx_x = threadIdx.x;\n    int __daisy_cuda_indvar_x = threadIdx.x + "
        "blockIdx.x*blockDim.x;\n    int __daisy_cuda_thread_idx_y = threadIdx.y;\n    int __daisy_cuda_indvar_y = "
        "threadIdx.y + blockIdx.y*blockDim.y;\n    int __daisy_cuda_thread_idx_z = threadIdx.z;\n    int "
        "__daisy_cuda_indvar_z = threadIdx.z + blockIdx.z*blockDim.z;\n    int i = __daisy_cuda_indvar_x;\n    int j = "
        "__daisy_cuda_indvar_y;\n    int k = __daisy_cuda_indvar_z;\n    if (i < 100) {\n            if (j < 40) {\n   "
        "                 if (k < 200) {\n                            {\n                                float in_ = "
        "0.0f;\n                                float out_;\n\n                                out_ = in_;\n\n         "
        "                       (reinterpret_cast<float *>(__daisy_cuda_A))[i + j + k] = out_;\n                       "
        "     }\n         "
        "           }\n            }\n    }\n}\n"
    );
}

TEST(CudaTransformTest, CudaTransformWithBlocksizeTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    auto& indvar = builder.add_container("i", int_desc);
    auto& jndvar = builder.add_container("j", int_desc);
    auto& kndvar = builder.add_container("k", int_desc);
    auto& A_device = builder.add_container("__daisy_cuda_A", pointer_type);
    auto& B_host = builder.add_container("B", base_desc);

    // Create a map with CUDA schedule
    ScheduleType cuda_schedule = ScheduleType_CUDA::create();
    ScheduleType_CUDA::dimension(cuda_schedule, CUDADimension::X);
    ScheduleType_CUDA::block_size(cuda_schedule, symbolic::integer(32));

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& map = builder.add_map(root, symbolic::symbol("i"), condition, init, update, cuda_schedule);

    analysis::AnalysisManager analysis_manager(builder.subject());

    // Create transform locally
    auto cuda_transform = CUDATransform(map, 64);
    if (cuda_transform.can_be_applied(builder, analysis_manager)) {
        cuda_transform.apply(builder, analysis_manager);
    }

    auto transformed_schedule = map.schedule_type();

    EXPECT_TRUE(SymEngine::eq(*ScheduleType_CUDA::block_size(transformed_schedule), *symbolic::integer(64)));
}


} // namespace sdfg::cuda
