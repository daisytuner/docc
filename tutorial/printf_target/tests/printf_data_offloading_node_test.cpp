#include <gtest/gtest.h>
#include <memory>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/function.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

#include "printf_data_offloading_node.h"
#include "printf_target.h"

namespace sdfg::printf_target {

TEST(PrintfDataOffloadingNode, AllocationDispatcherTest) {
    // Create a simple SDFG
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_desc);

    builder.add_container("A", pointer_type);

    auto& block = builder.add_block(root);

    // Create an allocation node
    auto& access_out = builder.add_access(block, "A");
    auto& alloc_node = builder.add_library_node<PrintfDataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(1024), // 1024 bytes
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC
    );

    builder.add_computational_memlet(block, alloc_node, "_ret", access_out, {}, pointer_type);

    // Create dispatcher and generate code
    codegen::CLanguageExtension language_extension(builder.subject());

    PrintfDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), alloc_node);

    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_code(stream, globals_stream, library_snippet_factory);

    std::string generated_code = stream.str();

    // Check that printf allocation message is generated
    EXPECT_NE(generated_code.find("[PRINTF_TARGET] Allocating"), std::string::npos);
    EXPECT_NE(generated_code.find("bytes"), std::string::npos);
    // Check that malloc is called (for simulation)
    EXPECT_NE(generated_code.find("malloc"), std::string::npos);
}

TEST(PrintfDataOffloadingNode, H2DTransferDispatcherTest) {
    // Create a simple SDFG
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_desc);

    builder.add_container("A_host", pointer_type);
    builder.add_container("A_device", pointer_type);

    auto& block = builder.add_block(root);

    // Create an H2D transfer node
    auto& access_host = builder.add_access(block, "A_host");
    auto& access_device = builder.add_access(block, "A_device");
    auto& h2d_node = builder.add_library_node<PrintfDataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(2048), // 2048 bytes
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE
    );

    builder.add_computational_memlet(block, access_host, h2d_node, "_src", {}, pointer_type);
    builder.add_computational_memlet(block, h2d_node, "_dst", access_device, {}, pointer_type);

    // Create dispatcher and generate code
    codegen::CLanguageExtension language_extension(builder.subject());

    PrintfDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), h2d_node);

    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_code(stream, globals_stream, library_snippet_factory);

    std::string generated_code = stream.str();

    // Check that printf H2D message is generated
    EXPECT_NE(generated_code.find("[PRINTF_TARGET] Copying"), std::string::npos);
    EXPECT_NE(generated_code.find("host"), std::string::npos);
    EXPECT_NE(generated_code.find("device"), std::string::npos);
    // Check that memcpy is called (for simulation)
    EXPECT_NE(generated_code.find("memcpy"), std::string::npos);
}

TEST(PrintfDataOffloadingNode, D2HTransferDispatcherTest) {
    // Create a simple SDFG
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_desc);

    builder.add_container("A_host", pointer_type);
    builder.add_container("A_device", pointer_type);

    auto& block = builder.add_block(root);

    // Create a D2H transfer node
    auto& access_device = builder.add_access(block, "A_device");
    auto& access_host = builder.add_access(block, "A_host");
    auto& d2h_node = builder.add_library_node<PrintfDataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(4096), // 4096 bytes
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE
    );

    builder.add_computational_memlet(block, access_device, d2h_node, "_src", {}, pointer_type);
    builder.add_computational_memlet(block, d2h_node, "_dst", access_host, {}, pointer_type);

    // Create dispatcher and generate code
    codegen::CLanguageExtension language_extension(builder.subject());

    PrintfDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), d2h_node);

    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_code(stream, globals_stream, library_snippet_factory);

    std::string generated_code = stream.str();

    // Check that printf D2H message is generated
    EXPECT_NE(generated_code.find("[PRINTF_TARGET] Copying"), std::string::npos);
    EXPECT_NE(generated_code.find("device"), std::string::npos);
    EXPECT_NE(generated_code.find("host"), std::string::npos);
}

TEST(PrintfDataOffloadingNode, FreeDispatcherTest) {
    // Create a simple SDFG
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_desc);

    builder.add_container("A", pointer_type);

    auto& block = builder.add_block(root);

    // Create a free node
    auto& access_in = builder.add_access(block, "A");
    auto& access_out = builder.add_access(block, "A");
    auto& free_node = builder.add_library_node<PrintfDataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(1024),
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE
    );

    builder.add_computational_memlet(block, access_in, free_node, "_ptr", {}, pointer_type);
    builder.add_computational_memlet(block, free_node, "_ptr", access_out, {}, pointer_type);

    // Create dispatcher and generate code
    codegen::CLanguageExtension language_extension(builder.subject());

    PrintfDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), free_node);

    codegen::PrettyPrinter stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_code(stream, globals_stream, library_snippet_factory);

    std::string generated_code = stream.str();

    // Check that printf free message is generated
    EXPECT_NE(generated_code.find("[PRINTF_TARGET] Freeing"), std::string::npos);
    // Check that free is called
    EXPECT_NE(generated_code.find("free"), std::string::npos);
}

TEST(PrintfDataOffloadingNode, CloneTest) {
    // Create a simple SDFG
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_desc);

    builder.add_container("A", pointer_type);

    auto& block = builder.add_block(root);

    // Create an allocation node
    auto& access_out = builder.add_access(block, "A");
    auto& alloc_node = builder.add_library_node<PrintfDataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(1024),
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC
    );

    // Test clone
    auto cloned = alloc_node.clone(999, graph::Vertex(), block.dataflow());
    auto& cloned_node = dynamic_cast<PrintfDataOffloadingNode&>(*cloned);

    EXPECT_EQ(cloned_node.transfer_direction(), offloading::DataTransferDirection::H2D);
    EXPECT_EQ(cloned_node.buffer_lifecycle(), offloading::BufferLifecycle::ALLOC);
    EXPECT_TRUE(symbolic::eq(cloned_node.size(), symbolic::integer(1024)));
}

TEST(PrintfDataOffloadingNode, ValidationTest) {
    // Create a simple SDFG
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_desc);

    builder.add_container("A", pointer_type);

    auto& block = builder.add_block(root);

    // Create a valid allocation node
    auto& access_out = builder.add_access(block, "A");
    auto& alloc_node = builder.add_library_node<PrintfDataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(1024),
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC
    );

    // Validation should not throw
    EXPECT_NO_THROW(alloc_node.validate(builder.subject()));
}

TEST(PrintfDataOffloadingNode, InstrumentationInfoTest) {
    // Create a simple SDFG
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_desc);

    builder.add_container("A_host", pointer_type);
    builder.add_container("A_device", pointer_type);

    auto& block = builder.add_block(root);

    // Create an H2D transfer node
    auto& access_host = builder.add_access(block, "A_host");
    auto& access_device = builder.add_access(block, "A_device");
    auto& h2d_node = builder.add_library_node<PrintfDataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(2048),
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE
    );

    builder.add_computational_memlet(block, access_host, h2d_node, "_src", {}, pointer_type);
    builder.add_computational_memlet(block, h2d_node, "_dst", access_device, {}, pointer_type);

    codegen::CLanguageExtension language_extension(builder.subject());

    PrintfDataOffloadingNodeDispatcher dispatcher(language_extension, builder.subject(), block.dataflow(), h2d_node);

    auto info = dispatcher.instrumentation_info();

    EXPECT_EQ(info.target_type().value(), TargetType_Printf.value());
    EXPECT_EQ(info.element_desc(), "h2d_transfer");
}

TEST(PrintfDataOffloadingNode, SerializationTest) {
    // Create a simple SDFG
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_desc);

    builder.add_container("A", pointer_type);

    auto& block = builder.add_block(root);

    // Create an H2D allocation node
    auto& access_out = builder.add_access(block, "A");
    auto& node = builder.add_library_node<PrintfDataOffloadingNode>(
        block,
        DebugInfo(),
        symbolic::integer(4096),
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC
    );

    builder.add_computational_memlet(block, node, "_dst", access_out, {}, pointer_type);

    // Test serialization
    PrintfDataOffloadingNodeSerializer serializer;
    nlohmann::json j = serializer.serialize(node);

    EXPECT_EQ(j["code"], LibraryNodeType_Printf_Offloading.value());
    EXPECT_EQ(j["transfer_direction"], static_cast<int8_t>(offloading::DataTransferDirection::H2D));
    EXPECT_EQ(j["buffer_lifecycle"], static_cast<int8_t>(offloading::BufferLifecycle::ALLOC));
}

TEST(PrintfDataOffloadingNode, LibraryNodeCodeTest) {
    EXPECT_EQ(LibraryNodeType_Printf_Offloading.value(), "PrintfOffloading");
}

} // namespace sdfg::printf_target
