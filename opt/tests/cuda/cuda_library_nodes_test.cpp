#include <gtest/gtest.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "symengine/symengine_rcp.h"

namespace sdfg::cuda {

TEST(CUDAD2HTransferTest, CloneTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);

    auto [block, d2h_transfer] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::integer(1024),
        symbolic::integer(0)
    );

    auto cloned_node = d2h_transfer.clone(1, d2h_transfer.vertex(), block.dataflow());

    ASSERT_TRUE(cloned_node != nullptr);
    ASSERT_TRUE(dynamic_cast<CUDADataOffloadingNode*>(cloned_node.get()) != nullptr);
    auto* cloned_node_ptr = dynamic_cast<CUDADataOffloadingNode*>(cloned_node.get());

    // EXPECT_EQ(cloned_node_ptr->element_id(), d2h_transfer.element_id());
    EXPECT_EQ(cloned_node_ptr->debug_info().filename(), "test_file.cpp");
    EXPECT_EQ(cloned_node_ptr->debug_info().start_line(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().end_line(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().start_column(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().end_column(), 10);
    EXPECT_TRUE(symbolic::eq(cloned_node_ptr->device_id(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(cloned_node_ptr->size(), symbolic::integer(1024)));
}

TEST(CUDAH2DTransferTest, CloneTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);

    auto [block, h2d_transfer] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::integer(1024),
        symbolic::integer(0)
    );

    auto cloned_node = h2d_transfer.clone(1, h2d_transfer.vertex(), block.dataflow());

    ASSERT_TRUE(cloned_node != nullptr);
    ASSERT_TRUE(dynamic_cast<CUDADataOffloadingNode*>(cloned_node.get()) != nullptr);
    auto* cloned_node_ptr = dynamic_cast<CUDADataOffloadingNode*>(cloned_node.get());

    // EXPECT_EQ(cloned_node_ptr->element_id(), h2d_transfer.element_id());
    EXPECT_EQ(cloned_node_ptr->debug_info().filename(), "test_file.cpp");
    EXPECT_EQ(cloned_node_ptr->debug_info().start_line(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().end_line(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().start_column(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().end_column(), 10);
    EXPECT_TRUE(symbolic::eq(cloned_node_ptr->device_id(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(cloned_node_ptr->size(), symbolic::integer(1024)));
}

TEST(CUDAMallocTest, CloneTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    auto& A_device = builder.add_container("A_device", pointer_type);

    auto [block, malloc_node] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_device",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::integer(1024),
        symbolic::integer(0)
    );

    auto cloned_node = malloc_node.clone(1, malloc_node.vertex(), block.dataflow());

    ASSERT_TRUE(cloned_node != nullptr);
    ASSERT_TRUE(dynamic_cast<CUDADataOffloadingNode*>(cloned_node.get()) != nullptr);
    auto* cloned_node_ptr = dynamic_cast<CUDADataOffloadingNode*>(cloned_node.get());

    // EXPECT_EQ(cloned_node_ptr->element_id(), malloc_node.element_id());
    EXPECT_EQ(cloned_node_ptr->debug_info().filename(), "test_file.cpp");
    EXPECT_EQ(cloned_node_ptr->debug_info().start_line(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().end_line(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().start_column(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().end_column(), 10);
    EXPECT_TRUE(symbolic::eq(cloned_node_ptr->device_id(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(cloned_node_ptr->size(), symbolic::integer(1024)));
}

TEST(CUDAFreeTest, CloneTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    auto& A_device = builder.add_container("A_device", pointer_type);

    auto [block, free_node] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_device",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        SymEngine::null,
        symbolic::integer(0)
    );

    auto cloned_node = free_node.clone(1, free_node.vertex(), block.dataflow());

    ASSERT_TRUE(cloned_node != nullptr);
    ASSERT_TRUE(dynamic_cast<CUDADataOffloadingNode*>(cloned_node.get()) != nullptr);
    auto* cloned_node_ptr = dynamic_cast<CUDADataOffloadingNode*>(cloned_node.get());

    // EXPECT_EQ(cloned_node_ptr->element_id(), free_node.element_id());
    EXPECT_EQ(cloned_node_ptr->debug_info().filename(), "test_file.cpp");
    EXPECT_EQ(cloned_node_ptr->debug_info().start_line(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().end_line(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().start_column(), 1);
    EXPECT_EQ(cloned_node_ptr->debug_info().end_column(), 10);
    EXPECT_TRUE(symbolic::eq(cloned_node_ptr->device_id(), symbolic::integer(0)));
}

TEST(CUDAD2HTransferTest, ReplaceTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, d2h_transfer] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    // Replace the size with a symbolic expression
    d2h_transfer.replace(symbolic::symbol("N"), symbolic::symbol("i"));
    // cast the node to CUDAD2HTransfer
    auto* real_d2h_transfer = dynamic_cast<CUDADataOffloadingNode*>(&d2h_transfer);
    EXPECT_TRUE(symbolic::eq(real_d2h_transfer->size(), symbolic::symbol("i")));
}

TEST(CUDAH2DTransferTest, ReplaceTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, h2d_transfer] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    // Replace the size with a symbolic expression
    h2d_transfer.replace(symbolic::symbol("N"), symbolic::symbol("i"));

    // cast the node to CUDAH2DTransfer
    auto* real_h2d_transfer = dynamic_cast<CUDADataOffloadingNode*>(&h2d_transfer);
    EXPECT_TRUE(symbolic::eq(real_h2d_transfer->size(), symbolic::symbol("i")));
}

TEST(CUDAMallocTest, ReplaceTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, malloc_node] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_device",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    // Replace the size with a symbolic expression
    malloc_node.replace(symbolic::symbol("N"), symbolic::symbol("i"));

    // cast the node to CUDAMalloc
    auto* real_malloc_node = dynamic_cast<CUDADataOffloadingNode*>(&malloc_node);
    EXPECT_TRUE(symbolic::eq(real_malloc_node->size(), symbolic::symbol("i")));
}

TEST(CUDAD2HTransferTest, SerializeDeserializeTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, d2h_transfer] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    auto& sdfg = builder.subject();
    serializer::JSONSerializer serializer;
    nlohmann::json j = serializer.serialize(sdfg);

    auto deserialized_sdfg = serializer.deserialize(j);

    EXPECT_TRUE(deserialized_sdfg != nullptr);

    EXPECT_TRUE(deserialized_sdfg->root().size() == 1);
    auto& des_block = deserialized_sdfg->root().at(0).first;
    auto& des_dataflow = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(des_block).dataflow();
    EXPECT_TRUE(des_dataflow.nodes().size() == 3);
    EXPECT_TRUE(des_dataflow.edges().size() == 2);
    bool found_d2h_transfer = false;
    for (const auto& node : des_dataflow.nodes()) {
        if (auto d2h_node = dynamic_cast<const CUDADataOffloadingNode*>(&node)) {
            found_d2h_transfer = true;
            EXPECT_EQ(d2h_node->debug_info().filename(), "test_file.cpp");
            EXPECT_EQ(d2h_node->debug_info().start_line(), 1);
            EXPECT_EQ(d2h_node->debug_info().end_line(), 1);
            EXPECT_EQ(d2h_node->debug_info().start_column(), 1);
            EXPECT_EQ(d2h_node->debug_info().end_column(), 10);
            EXPECT_TRUE(symbolic::eq(d2h_node->device_id(), symbolic::integer(0)));
            EXPECT_TRUE(symbolic::eq(d2h_node->size(), symbolic::symbol("N")));
        }
    }
    EXPECT_TRUE(found_d2h_transfer);
}

TEST(CUDAH2DTransferTest, SerializeDeserializeTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, h2d_transfer] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    auto& sdfg = builder.subject();
    serializer::JSONSerializer serializer;
    nlohmann::json j = serializer.serialize(sdfg);

    auto deserialized_sdfg = serializer.deserialize(j);

    EXPECT_TRUE(deserialized_sdfg != nullptr);

    EXPECT_TRUE(deserialized_sdfg->root().size() == 1);
    auto& des_block = deserialized_sdfg->root().at(0).first;
    auto& des_dataflow = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(des_block).dataflow();
    EXPECT_TRUE(des_dataflow.nodes().size() == 3);
    EXPECT_TRUE(des_dataflow.edges().size() == 2);

    bool found_h2d_transfer = false;
    for (const auto& node : des_dataflow.nodes()) {
        if (auto h2d_node = dynamic_cast<const CUDADataOffloadingNode*>(&node)) {
            found_h2d_transfer = true;
            EXPECT_EQ(h2d_node->element_id(), h2d_transfer.element_id());
            EXPECT_EQ(h2d_node->debug_info().filename(), "test_file.cpp");
            EXPECT_EQ(h2d_node->debug_info().start_line(), 1);
            EXPECT_EQ(h2d_node->debug_info().end_line(), 1);
            EXPECT_EQ(h2d_node->debug_info().start_column(), 1);
            EXPECT_EQ(h2d_node->debug_info().end_column(), 10);
            EXPECT_TRUE(symbolic::eq(h2d_node->device_id(), symbolic::integer(0)));
            EXPECT_TRUE(symbolic::eq(h2d_node->size(), symbolic::symbol("N")));
        }
    }
    EXPECT_TRUE(found_h2d_transfer);
}

TEST(CUDAMallocTest, SerializeDeserializeTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, malloc_node] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_device",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    auto& sdfg = builder.subject();
    serializer::JSONSerializer serializer;
    nlohmann::json j = serializer.serialize(sdfg);

    auto deserialized_sdfg = serializer.deserialize(j);

    EXPECT_TRUE(deserialized_sdfg != nullptr);

    EXPECT_TRUE(deserialized_sdfg->root().size() == 1);
    auto& des_block = deserialized_sdfg->root().at(0).first;
    auto& des_dataflow = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(des_block).dataflow();
    EXPECT_TRUE(des_dataflow.nodes().size() == 2);
    EXPECT_TRUE(des_dataflow.edges().size() == 1);

    bool found_malloc_node = false;
    for (const auto& node : des_dataflow.nodes()) {
        if (auto malloc_node_ptr = dynamic_cast<const CUDADataOffloadingNode*>(&node)) {
            found_malloc_node = true;
            EXPECT_EQ(malloc_node_ptr->element_id(), malloc_node.element_id());
            EXPECT_EQ(malloc_node_ptr->debug_info().filename(), "test_file.cpp");
            EXPECT_EQ(malloc_node_ptr->debug_info().start_line(), 1);
            EXPECT_EQ(malloc_node_ptr->debug_info().end_line(), 1);
            EXPECT_EQ(malloc_node_ptr->debug_info().start_column(), 1);
            EXPECT_EQ(malloc_node_ptr->debug_info().end_column(), 10);
            EXPECT_TRUE(symbolic::eq(malloc_node_ptr->device_id(), symbolic::integer(0)));
            EXPECT_TRUE(symbolic::eq(malloc_node_ptr->size(), symbolic::symbol("N")));
        }
    }
    EXPECT_TRUE(found_malloc_node);
}

TEST(CUDAFreeTest, SerializeDeserializeTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    auto& A_device = builder.add_container("A_device", pointer_type);

    auto [block, free_node] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_device",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        SymEngine::null,
        symbolic::integer(0)
    );

    auto& sdfg = builder.subject();
    serializer::JSONSerializer serializer;
    nlohmann::json j = serializer.serialize(sdfg);

    auto deserialized_sdfg = serializer.deserialize(j);

    EXPECT_TRUE(deserialized_sdfg != nullptr);

    EXPECT_TRUE(deserialized_sdfg->root().size() == 1);
    auto& des_block = deserialized_sdfg->root().at(0).first;
    auto& des_dataflow = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(des_block).dataflow();
    EXPECT_EQ(des_dataflow.nodes().size(), 2);
    EXPECT_EQ(des_dataflow.edges().size(), 1);

    bool found_free_node = false;
    for (const auto& node : des_dataflow.nodes()) {
        if (auto free_node_ptr = dynamic_cast<const CUDADataOffloadingNode*>(&node)) {
            found_free_node = true;
            EXPECT_EQ(free_node_ptr->element_id(), free_node.element_id());
            EXPECT_EQ(free_node_ptr->debug_info().filename(), "test_file.cpp");
            EXPECT_EQ(free_node_ptr->debug_info().start_line(), 1);
            EXPECT_EQ(free_node_ptr->debug_info().end_line(), 1);
            EXPECT_EQ(free_node_ptr->debug_info().start_column(), 1);
            EXPECT_EQ(free_node_ptr->debug_info().end_column(), 10);
            EXPECT_TRUE(symbolic::eq(free_node_ptr->device_id(), symbolic::integer(0)));
        }
    }
    EXPECT_TRUE(found_free_node);
}

TEST(CUDAD2HTransferTest, DispatcherTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, d2h_node] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    // Create a dispatcher for the CUDAD2HTransfer node
    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    CUDADataOffloadingNodeDispatcher
        dispatcher_instance(language_extension, builder.subject(), block.dataflow(), d2h_node);
    dispatcher_instance.dispatch(pretty_printer, globals_printer, snippet_factory);

    // Check if the generated code contains the expected function call
    std::string expected_code = R"({
    cudaError_t err;
    err = cudaMemcpy(((float *) A_host), ((float *) A_device), N, cudaMemcpyDeviceToHost);
)"
                                "}\n";
    std::string generated_code = pretty_printer.str();
    EXPECT_EQ(expected_code, generated_code);
}

TEST(CUDAH2DTransferTest, DispatcherTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, h2d_transfer] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    // Create a dispatcher for the CUDAH2DTransfer node
    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    CUDADataOffloadingNodeDispatcher
        dispatcher_instance(language_extension, builder.subject(), block.dataflow(), h2d_transfer);
    dispatcher_instance.dispatch(pretty_printer, globals_printer, snippet_factory);

    // Check if the generated code contains the expected function call
    std::string expected_code = R"({
    cudaError_t err;
    err = cudaMemcpy(((float *) A_device), ((float *) A_host), N, cudaMemcpyHostToDevice);
)"
                                "}\n";
    std::string generated_code = pretty_printer.str();
    EXPECT_EQ(expected_code, generated_code);
}

TEST(CUDAMallocTest, DispatcherTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, malloc_node] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_device",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    // Create a dispatcher for the CUDAMalloc node
    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    CUDADataOffloadingNodeDispatcher
        dispatcher_instance(language_extension, builder.subject(), block.dataflow(), malloc_node);
    dispatcher_instance.dispatch(pretty_printer, globals_printer, snippet_factory);

    // Check if the generated code contains the expected function call
    std::string expected_code = R"({
    cudaError_t err;
    float *_dev;
    err = cudaMalloc(&_dev, N);
    A_device = _dev;
)"
                                "}\n";
    std::string generated_code = pretty_printer.str();
    EXPECT_EQ(expected_code, generated_code);
}

TEST(CUDAFreeTest, DispatcherTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    auto& A_device = builder.add_container("A_device", pointer_type);

    auto [block, free_node] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        SymEngine::null,
        symbolic::integer(0)
    );

    // Create a dispatcher for the CUDAFree node
    codegen::CLanguageExtension language_extension(builder.subject());
    codegen::PrettyPrinter pretty_printer;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;

    CUDADataOffloadingNodeDispatcher
        dispatcher_instance(language_extension, builder.subject(), block.dataflow(), free_node);
    dispatcher_instance.dispatch(pretty_printer, globals_printer, snippet_factory);

    // Check if the generated code contains the expected function call
    std::string expected_code = R"({
    cudaError_t err;
    err = cudaFree(((float *) A_device));
)"
                                "}\n";
    std::string generated_code = pretty_printer.str();
    EXPECT_EQ(expected_code, generated_code);
}

TEST(CUDAD2HTransferTest, SymbolSetTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, d2h_transfer] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    // Create a symbol set for the CUDAD2HTransfer node
    auto* real_d2h_transfer = dynamic_cast<CUDADataOffloadingNode*>(&d2h_transfer);
    ASSERT_TRUE(real_d2h_transfer != nullptr);
    auto symbol_set = real_d2h_transfer->symbols();

    EXPECT_TRUE(symbol_set.size() == 1);
    EXPECT_TRUE(symbol_set.begin()->get()->get_name() == "N");
}

TEST(CUDAH2DTransferTest, SymbolSetTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, h2d_transfer] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    // Create a symbol set for the CUDAH2DTransfer node
    auto* real_h2d_transfer = dynamic_cast<CUDADataOffloadingNode*>(&h2d_transfer);
    ASSERT_TRUE(real_h2d_transfer != nullptr);
    auto symbol_set = real_h2d_transfer->symbols();

    EXPECT_TRUE(symbol_set.size() == 1);
    EXPECT_TRUE(symbol_set.begin()->get()->get_name() == "N");
}

TEST(CUDAMallocTest, SymbolSetTest) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Scalar integer_desc(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_desc);

    auto& A_host = builder.add_container("A_host", pointer_type);
    auto& A_device = builder.add_container("A_device", pointer_type);
    auto& N = builder.add_container("N", integer_desc);
    auto& i = builder.add_container("i", integer_desc);

    auto [block, cuda_malloc] = offloading::add_offloading_block<CUDADataOffloadingNode>(
        builder,
        root,
        "A_host",
        "A_device",
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        pointer_type,
        DebugInfo("test_file.cpp", 1, 1, 1, 10),
        symbolic::symbol("N"),
        symbolic::integer(0)
    );

    // Create a symbol set for the CUDAMalloc node
    auto* real_cuda_malloc = dynamic_cast<CUDADataOffloadingNode*>(&cuda_malloc);
    ASSERT_TRUE(real_cuda_malloc != nullptr);
    auto symbol_set = real_cuda_malloc->symbols();

    EXPECT_TRUE(symbol_set.size() == 1);
    EXPECT_TRUE(symbol_set.begin()->get()->get_name() == "N");
}

TEST(CUDAScheduleTypeTest, ScheduleTypeTest) {
    ScheduleType cuda_schedule = ScheduleType_CUDA::create();

    EXPECT_EQ(cuda_schedule.value(), ScheduleType_CUDA::value());
    EXPECT_EQ(ScheduleType_CUDA::dimension(cuda_schedule), CUDADimension::X);

    ScheduleType_CUDA::dimension(cuda_schedule, CUDADimension::Y);

    serializer::JSONSerializer serializer;
    nlohmann::json j;
    serializer.schedule_type_to_json(j, cuda_schedule);

    EXPECT_EQ(ScheduleType_CUDA::dimension(cuda_schedule), CUDADimension::Y);

    ScheduleType_CUDA::block_size(cuda_schedule, symbolic::integer(256));
    EXPECT_TRUE(symbolic::eq(ScheduleType_CUDA::block_size(cuda_schedule), symbolic::integer(256)));
}

} // namespace sdfg::cuda
