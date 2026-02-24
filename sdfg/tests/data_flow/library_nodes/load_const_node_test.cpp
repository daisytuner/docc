#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/load_const_node.h"

#include <sdfg/codegen/code_generators/cpp_code_generator.h>
#include <sdfg/data_flow/library_nodes/stdlib/malloc.h>
#include <sdfg/visualizer/dot_visualizer.h>
using namespace sdfg;

TEST(LoadConstNodeTest, BuildInMemory2Code) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto constPtrType = types::Pointer(types::Scalar(types::PrimitiveType::Float));
    auto opaquePtrType = types::Pointer();
    auto resContName = "result";
    builder.add_container(resContName, opaquePtrType);

    auto& block = builder.add_block(builder.subject().root());
    auto arrPtr = constPtrType.clone();
    std::vector<float> data = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                               12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f};
    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(data.data());
    std::unique_ptr<data_flow::ConstSource> constSource =
        std::make_unique<data_flow::InMemoryConstSource>(data_ptr, data_ptr + data.size() * sizeof(float));
    auto& libNode =
        builder
            .add_library_node<data_flow::LoadConstNode>(block, DebugInfo(), std::move(arrPtr), std::move(constSource));
    auto& accNode = builder.add_access(block, "result");
    builder.add_computational_memlet(block, libNode, "_out", accNode, {}, constPtrType, DebugInfo());

    builder.add_return(builder.subject().root(), "result");

    auto sdfg = builder.move();
    sdfg->validate();

    visualizer::DotVisualizer::writeToFile(*sdfg);

    analysis::AnalysisManager ana(*sdfg);
    auto inst_plan = codegen::InstrumentationPlan::none(*sdfg);
    auto cap_plan = codegen::ArgCapturePlan::none(*sdfg);
    auto snippet_mgr = std::make_shared<codegen::CodeSnippetFactory>();
    codegen::CPPCodeGenerator codeGen(*sdfg, ana, *inst_plan, *cap_plan, snippet_mgr);
    codeGen.generate();

    auto expected_def =
        "static const uint8_t daisy_load_const_3[] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80, 0x3f, 0x0, 0x0, 0x0, 0x40, "
        "0x0, 0x0, 0x40, 0x40, 0x0, 0x0, 0x80, 0x40, 0x0, 0x0, 0xa0, 0x40, 0x0, 0x0, 0xc0, 0x40, 0x0, 0x0, 0xe0, 0x40, "
        "0x0, 0x0, 0x0, 0x41, 0x0, 0x0, 0x10, 0x41, 0x0, 0x0, 0x20, 0x41, 0x0, 0x0, 0x30, 0x41, 0x0, 0x0, 0x40, 0x41, "
        "0x0, 0x0, 0x50, 0x41, 0x0, 0x0, 0x60, 0x41, 0x0, 0x0, 0x70, 0x41, 0x0, 0x0, 0x80, 0x41, 0x0, 0x0, 0x88, 0x41, "
        "0x0, 0x0, 0x90, 0x41, 0x0, 0x0, 0x98, 0x41, 0x0, 0x0, 0xa0, 0x41, 0x0, 0x0, 0xa8, 0x41, 0x0, 0x0, 0xb0, 0x41, "
        "0x0, 0x0, 0xb8, 0x41};\n";
    EXPECT_EQ(codeGen.globals().str(), expected_def);

    auto expected_main = R"a(void* result;
    {
        float *_out = (reinterpret_cast<float *>(result));

        _out = const_cast<float *>(reinterpret_cast<const float *>(&daisy_load_const_3[0]));

        result = _out;
    }
    return result;
)a";
    EXPECT_EQ(codeGen.main().str(), expected_main);
}

TEST(LoadConstNodeTest, InMemorySer) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);

    auto constPtrType = types::Pointer(types::Scalar(types::PrimitiveType::Float));
    auto opaquePtrType = types::Pointer();
    auto resContName = "result";
    builder.add_container(resContName, opaquePtrType);

    auto& block = builder.add_block(builder.subject().root());
    auto arrPtr = constPtrType.clone();
    std::vector<float> data = {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                               12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f};
    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(data.data());
    std::unique_ptr<data_flow::ConstSource> constSource =
        std::make_unique<data_flow::InMemoryConstSource>(data_ptr, data_ptr + data.size() * sizeof(float));
    auto& libNode =
        builder
            .add_library_node<data_flow::LoadConstNode>(block, DebugInfo(), std::move(arrPtr), std::move(constSource));
    auto& accNode = builder.add_access(block, "result");
    builder.add_computational_memlet(block, libNode, "_out", accNode, {}, constPtrType, DebugInfo());

    builder.add_return(builder.subject().root(), "result");

    auto sdfg = builder.move();

    serializer::JSONSerializer ser;
    nlohmann::json j;
    ser.dataflow_to_json(j, block.dataflow());
    auto reimport = nlohmann::json::parse(j.dump(2));
    std::cout << j.dump(2) << std::endl;
    builder::StructuredSDFGBuilder builder2("reimport", FunctionType_CPU);
    builder2.add_container("result", opaquePtrType);
    auto& reimport_block = builder2.add_block(builder2.subject().root());
    ser.json_to_dataflow(reimport, builder2, reimport_block);
    auto& reimport_libNode = **reimport_block.dataflow().library_nodes().begin();
    EXPECT_EQ(libNode.code(), reimport_libNode.code());
    auto& org_source = dynamic_cast<data_flow::LoadConstNode&>(libNode).data_source();
    auto& new_source = dynamic_cast<data_flow::LoadConstNode&>(reimport_libNode).data_source();
    EXPECT_EQ(org_source.num_bytes(), new_source.num_bytes());

    EXPECT_EQ(
        std::vector<uint8_t>(org_source.inline_begin(), org_source.inline_end()),
        std::vector<uint8_t>(new_source.inline_begin(), new_source.inline_end())
    );
}
