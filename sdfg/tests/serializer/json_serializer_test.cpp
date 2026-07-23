#include "sdfg/serializer/json_serializer.h"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/element.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/function.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "symengine/expression.h"

using namespace sdfg;

TEST(JSONSerializerTest, DatatypeToJSON_Scalar) {
    // Create a sample data type
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, scalar_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "scalar");
    EXPECT_TRUE(j.contains("primitive_type"));
    EXPECT_EQ(j["primitive_type"], scalar_type.primitive_type());
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"]["value"].get<std::string>(), scalar_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], scalar_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], scalar_type.initializer());
}

TEST(JSONSerializerTest, DatatypeToJSON_Pointer) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, pointer_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "pointer");
    EXPECT_TRUE(j.contains("pointee_type"));
    EXPECT_EQ(j["pointee_type"]["type"], "scalar");
    EXPECT_EQ(j["pointee_type"]["primitive_type"], base_desc.primitive_type());
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"]["value"].get<std::string>(), pointer_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], pointer_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], pointer_type.initializer());
}

TEST(JSONSerializerTest, DatatypeToJSON_Structure) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Structure structure_type("MyStruct");
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, structure_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "structure");
    EXPECT_TRUE(j.contains("name"));
    EXPECT_EQ(j["name"], "MyStruct");
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"]["value"].get<std::string>(), structure_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], structure_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], structure_type.initializer());
}

TEST(JSONSerializerTest, DatatypeToJSON_Array) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_type(base_desc, {symbolic::symbol("N")});
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, array_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "array");
    EXPECT_TRUE(j.contains("element_type"));
    EXPECT_EQ(j["element_type"]["type"], "scalar");
    EXPECT_EQ(j["element_type"]["primitive_type"], base_desc.primitive_type());
    EXPECT_TRUE(j.contains("num_elements"));
    EXPECT_TRUE(symbolic::eq(SymEngine::Expression(j["num_elements"]), symbolic::symbol("N")));
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"]["value"].get<std::string>(), array_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], array_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], array_type.initializer());
}

TEST(JSONSerializerTest, DatatypeReferenceToJSON_Scalar) {
    // Create a sample data type
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    nlohmann::json j;

    sdfg::codegen::Reference reference_type(scalar_type);

    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, reference_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "reference");
    EXPECT_TRUE(j.contains("reference_type"));
    EXPECT_TRUE(j["reference_type"].contains("type"));
    EXPECT_EQ(j["reference_type"]["type"], "scalar");
    EXPECT_TRUE(j["reference_type"].contains("primitive_type"));
    EXPECT_EQ(j["reference_type"]["primitive_type"], scalar_type.primitive_type());
    EXPECT_TRUE(j["reference_type"].contains("storage_type"));
    EXPECT_EQ(j["reference_type"]["storage_type"]["value"].get<std::string>(), scalar_type.storage_type().value());
    EXPECT_TRUE(j["reference_type"].contains("alignment"));
    EXPECT_EQ(j["reference_type"]["alignment"], scalar_type.alignment());
    EXPECT_TRUE(j["reference_type"].contains("initializer"));
    EXPECT_EQ(j["reference_type"]["initializer"], scalar_type.initializer());
}

TEST(JSONSerializerTest, JSONToDatatypeReference_Scalar) {
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    sdfg::codegen::Reference reference_type(scalar_type);

    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    serializer.type_to_json(j, reference_type);

    auto deserialized = serializer.json_to_type(j);
    ASSERT_TRUE(deserialized != nullptr);
    ASSERT_TRUE(dynamic_cast<sdfg::codegen::Reference*>(deserialized.get()) != nullptr);

    auto* ref = dynamic_cast<sdfg::codegen::Reference*>(deserialized.get());
    EXPECT_TRUE(reference_type == *deserialized);
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&ref->reference_type()) != nullptr);
    auto& inner = dynamic_cast<const types::Scalar&>(ref->reference_type());
    EXPECT_EQ(inner.primitive_type(), scalar_type.primitive_type());
}

TEST(JSONSerializerTest, DatatypeReferenceToJSON_Pointer) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    nlohmann::json j;

    sdfg::codegen::Reference reference_type(pointer_type);

    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, reference_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "reference");
    EXPECT_TRUE(j.contains("reference_type"));
    EXPECT_TRUE(j["reference_type"].contains("type"));
    EXPECT_EQ(j["reference_type"]["type"], "pointer");
    EXPECT_TRUE(j["reference_type"].contains("pointee_type"));
    EXPECT_EQ(j["reference_type"]["pointee_type"]["type"], "scalar");
    EXPECT_EQ(j["reference_type"]["pointee_type"]["primitive_type"], base_desc.primitive_type());
    EXPECT_TRUE(j["reference_type"].contains("storage_type"));
    EXPECT_EQ(j["reference_type"]["storage_type"]["value"].get<std::string>(), pointer_type.storage_type().value());
    EXPECT_TRUE(j["reference_type"].contains("alignment"));
    EXPECT_EQ(j["reference_type"]["alignment"], pointer_type.alignment());
    EXPECT_TRUE(j["reference_type"].contains("initializer"));
    EXPECT_EQ(j["reference_type"]["initializer"], pointer_type.initializer());
}

TEST(JSONSerializerTest, JSONToDatatypeReference_Pointer) {
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);
    sdfg::codegen::Reference reference_type(pointer_type);

    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    serializer.type_to_json(j, reference_type);

    auto deserialized = serializer.json_to_type(j);
    ASSERT_TRUE(deserialized != nullptr);
    ASSERT_TRUE(dynamic_cast<sdfg::codegen::Reference*>(deserialized.get()) != nullptr);

    auto* ref = dynamic_cast<sdfg::codegen::Reference*>(deserialized.get());
    EXPECT_TRUE(reference_type == *deserialized);
    EXPECT_TRUE(dynamic_cast<const types::Pointer*>(&ref->reference_type()) != nullptr);
    auto& inner = dynamic_cast<const types::Pointer&>(ref->reference_type());
    EXPECT_TRUE(inner.has_pointee_type());
    EXPECT_EQ(inner.pointee_type().primitive_type(), base_desc.primitive_type());
}

TEST(JSONSerializerTest, DatatypeReferenceToJSON_Structure) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Structure structure_type("MyStruct");
    nlohmann::json j;

    sdfg::serializer::JSONSerializer serializer;
    sdfg::codegen::Reference reference_type(structure_type);

    // Serialize the data type to JSON
    serializer.type_to_json(j, reference_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "reference");
    EXPECT_TRUE(j.contains("reference_type"));
    EXPECT_TRUE(j["reference_type"].contains("type"));
    EXPECT_EQ(j["reference_type"]["type"], "structure");
    EXPECT_TRUE(j["reference_type"].contains("name"));
    EXPECT_EQ(j["reference_type"]["name"], "MyStruct");
    EXPECT_TRUE(j["reference_type"].contains("storage_type"));
    EXPECT_EQ(j["reference_type"]["storage_type"]["value"].get<std::string>(), structure_type.storage_type().value());
    EXPECT_TRUE(j["reference_type"].contains("alignment"));
    EXPECT_EQ(j["reference_type"]["alignment"], structure_type.alignment());
    EXPECT_TRUE(j["reference_type"].contains("initializer"));
    EXPECT_EQ(j["reference_type"]["initializer"], structure_type.initializer());
}

TEST(JSONSerializerTest, JSONToDatatypeReference_Structure) {
    types::Structure structure_type("MyStruct");
    sdfg::codegen::Reference reference_type(structure_type);

    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    serializer.type_to_json(j, reference_type);

    auto deserialized = serializer.json_to_type(j);
    ASSERT_TRUE(deserialized != nullptr);
    ASSERT_TRUE(dynamic_cast<sdfg::codegen::Reference*>(deserialized.get()) != nullptr);

    auto* ref = dynamic_cast<sdfg::codegen::Reference*>(deserialized.get());
    EXPECT_TRUE(reference_type == *deserialized);
    EXPECT_TRUE(dynamic_cast<const types::Structure*>(&ref->reference_type()) != nullptr);
    auto& inner = dynamic_cast<const types::Structure&>(ref->reference_type());
    EXPECT_EQ(inner.name(), structure_type.name());
}

TEST(JSONSerializerTest, DatatypeReferenceToJSON_Array) {
    // Create a sample data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_type(base_desc, {symbolic::symbol("N")});
    nlohmann::json j;
    sdfg::codegen::Reference reference_type(array_type);

    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, reference_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "reference");
    EXPECT_TRUE(j.contains("reference_type"));
    EXPECT_TRUE(j["reference_type"].contains("type"));
    EXPECT_EQ(j["reference_type"]["type"], "array");
    EXPECT_EQ(j["reference_type"]["element_type"]["type"], "scalar");
    EXPECT_EQ(j["reference_type"]["element_type"]["primitive_type"], base_desc.primitive_type());
    EXPECT_TRUE(j["reference_type"].contains("num_elements"));
    EXPECT_TRUE(symbolic::eq(SymEngine::Expression(j["reference_type"]["num_elements"]), symbolic::symbol("N")));
    EXPECT_TRUE(j["reference_type"].contains("storage_type"));
    EXPECT_EQ(j["reference_type"]["storage_type"]["value"].get<std::string>(), array_type.storage_type().value());
    EXPECT_TRUE(j["reference_type"].contains("alignment"));
    EXPECT_EQ(j["reference_type"]["alignment"], array_type.alignment());
    EXPECT_TRUE(j["reference_type"].contains("initializer"));
    EXPECT_EQ(j["reference_type"]["initializer"], array_type.initializer());
}

TEST(JSONSerializerTest, JSONToDatatypeReference_Array) {
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_type(base_desc, {symbolic::symbol("N")});
    sdfg::codegen::Reference reference_type(array_type);

    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    serializer.type_to_json(j, reference_type);

    auto deserialized = serializer.json_to_type(j);
    ASSERT_TRUE(deserialized != nullptr);
    ASSERT_TRUE(dynamic_cast<sdfg::codegen::Reference*>(deserialized.get()) != nullptr);

    auto* ref = dynamic_cast<sdfg::codegen::Reference*>(deserialized.get());
    EXPECT_TRUE(reference_type == *deserialized);
    EXPECT_TRUE(dynamic_cast<const types::Array*>(&ref->reference_type()) != nullptr);
    auto& inner = dynamic_cast<const types::Array&>(ref->reference_type());
    EXPECT_EQ(inner.element_type().primitive_type(), base_desc.primitive_type());
    EXPECT_TRUE(sdfg::symbolic::eq(inner.num_elements(), array_type.num_elements()));
}

TEST(JSONSerializerTest, DatatypeToJSON_Function) {
    // Create a sample data type
    types::Scalar scalar_type(types::PrimitiveType::Int32);
    types::Function function_type(scalar_type, true);
    function_type.add_param(scalar_type);
    nlohmann::json j;

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, function_type);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "function");
    EXPECT_TRUE(j.contains("return_type"));
    EXPECT_EQ(j["return_type"]["type"], "scalar");
    EXPECT_EQ(j["return_type"]["primitive_type"], scalar_type.primitive_type());
    EXPECT_TRUE(j.contains("params"));
    EXPECT_EQ(j["params"].size(), 1);
    EXPECT_EQ(j["params"][0]["type"], "scalar");
    EXPECT_EQ(j["params"][0]["primitive_type"], scalar_type.primitive_type());
    EXPECT_TRUE(j.contains("is_var_arg"));
    EXPECT_EQ(j["is_var_arg"], function_type.is_var_arg());
    EXPECT_TRUE(j.contains("storage_type"));
    EXPECT_EQ(j["storage_type"]["value"].get<std::string>(), scalar_type.storage_type().value());
    EXPECT_TRUE(j.contains("alignment"));
    EXPECT_EQ(j["alignment"], scalar_type.alignment());
    EXPECT_TRUE(j.contains("initializer"));
    EXPECT_EQ(j["initializer"], scalar_type.initializer());
}

TEST(JSONSerializerTest, DataflowToJSON) {
    // Create a sample Block object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc, true);
    builder.add_container("D", opaque_desc, true);
    builder.add_container("C", base_desc, true);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);

    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "D");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    auto& memlet_in =
        builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, desc);
    auto& memlet_in2 = builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {});
    auto& memlet_out =
        builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")}, desc);

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Block to JSON
    nlohmann::json j;
    serializer.dataflow_to_json(j, block.dataflow());

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "dataflow");
    EXPECT_TRUE(j.contains("nodes"));
    EXPECT_TRUE(j.contains("edges"));
    EXPECT_EQ(j["nodes"].size(), 4);
    EXPECT_EQ(j["edges"].size(), 3);

    // Check if the nodes and edges are serialized correctly
    auto access_node_A = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "access_node" && node["data"] == "A";
    });
    auto access_node_C = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "access_node" && node["data"] == "C";
    });
    auto access_node_D = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "access_node" && node["data"] == "D";
    });
    auto tasklet_node = std::find_if(j["nodes"].begin(), j["nodes"].end(), [](const auto& node) {
        return node["type"] == "tasklet" && node["code"] == data_flow::TaskletCode::fp_add;
    });

    EXPECT_NE(access_node_A, j["nodes"].end());
    EXPECT_NE(access_node_C, j["nodes"].end());
    EXPECT_NE(access_node_D, j["nodes"].end());
    EXPECT_NE(tasklet_node, j["nodes"].end());

    EXPECT_EQ(tasklet_node->at("inputs").size(), 2);
    EXPECT_EQ(tasklet_node->at("output"), "_out");

    auto edge_to_tasklet = std::find_if(j["edges"].begin(), j["edges"].end(), [&](const auto& edge) {
        return edge["src"] == access_node_A->at("element_id") && edge["dst"] == tasklet_node->at("element_id") &&
               edge["src_conn"] == "void" && edge["dst_conn"] == "_in1";
    });
    auto edge_to_tasklet2 = std::find_if(j["edges"].begin(), j["edges"].end(), [&](const auto& edge) {
        return edge["src"] == access_node_C->at("element_id") && edge["dst"] == tasklet_node->at("element_id") &&
               edge["src_conn"] == "void" && edge["dst_conn"] == "_in2";
    });
    auto edge_from_tasklet = std::find_if(j["edges"].begin(), j["edges"].end(), [&](const auto& edge) {
        return edge["src"] == tasklet_node->at("element_id") && edge["dst"] == access_node_D->at("element_id") &&
               edge["src_conn"] == "_out" && edge["dst_conn"] == "void";
    });

    EXPECT_NE(edge_to_tasklet, j["edges"].end());
    EXPECT_NE(edge_to_tasklet2, j["edges"].end());
    EXPECT_NE(edge_from_tasklet, j["edges"].end());
}

TEST(JSONSerializerTest, BlockToJSON) {
    // Create a sample Block object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Block to JSON
    nlohmann::json j;
    serializer.block_to_json(j, block);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "block");
}

TEST(JSONSerializerTest, ForNodeToJSON) {
    // Create a sample For node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = builder.add_block(scope.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the For node to JSON
    nlohmann::json j;
    serializer.structured_loop_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "for");

    EXPECT_TRUE(j.contains("indvar"));
    EXPECT_EQ(j["indvar"], "i");
    EXPECT_TRUE(j.contains("condition"));
    EXPECT_EQ(j["condition"], "(i < 10)");
    EXPECT_TRUE(j.contains("init"));
    EXPECT_EQ(j["init"], "0");
    EXPECT_TRUE(j.contains("update"));
    EXPECT_TRUE(symbolic::
                    eq(SymEngine::Expression(j["update"]), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));
    EXPECT_TRUE(j.contains("root"));
    EXPECT_EQ(j["root"]["type"], "sequence");
    EXPECT_EQ(j["root"]["children"].size(), 1);
    EXPECT_EQ(j["root"]["children"][0]["type"], "block");
}

TEST(JSONSerializerTest, IfElseToJSON) {
    // Create a sample IfElse node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& if_else = builder.add_if_else(root);
    auto& true_case = builder.add_case(if_else, symbolic::__true__());
    auto& false_case = builder.add_case(if_else, symbolic::__false__());
    auto& true_block = builder.add_block(true_case);
    auto& false_block = builder.add_block(false_case);

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the IfElse node to JSON
    nlohmann::json j;
    serializer.if_else_to_json(j, if_else);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "if_else");

    EXPECT_TRUE(j.contains("branches"));
    EXPECT_EQ(j["branches"].size(), 2);
    EXPECT_EQ(j["branches"][0]["condition"], "True");
    EXPECT_EQ(j["branches"][1]["condition"], "False");
    EXPECT_EQ(j["branches"][0]["root"]["type"], "sequence");
    EXPECT_EQ(j["branches"][0]["root"]["children"].size(), 1);
    EXPECT_EQ(j["branches"][0]["root"]["children"][0]["type"], "block");
    EXPECT_EQ(j["branches"][1]["root"]["type"], "sequence");
    EXPECT_EQ(j["branches"][1]["root"]["children"].size(), 1);
    EXPECT_EQ(j["branches"][1]["root"]["children"][0]["type"], "block");
}

TEST(JSONSerializerTest, WhileToJSON_break) {
    // Create a sample While node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_while(root);
    auto& body = builder.add_block(scope.root());
    auto& break_state = builder.add_break(scope.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the While node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "while");

    EXPECT_TRUE(j.contains("root"));
    EXPECT_EQ(j["root"]["type"], "sequence");
    EXPECT_EQ(j["root"]["children"].size(), 2);
    EXPECT_EQ(j["root"]["children"][0]["type"], "block");
    EXPECT_EQ(j["root"]["children"][1]["type"], "break");
}

TEST(JSONSerializerTest, WhileToJSON_continue) {
    // Create a sample While node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_while(root);
    auto& body = builder.add_block(scope.root());
    auto& continue_state = builder.add_continue(scope.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the While node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "while");

    EXPECT_TRUE(j.contains("root"));
    EXPECT_EQ(j["root"]["type"], "sequence");
    EXPECT_EQ(j["root"]["children"].size(), 2);
    EXPECT_EQ(j["root"]["children"][0]["type"], "block");
    EXPECT_EQ(j["root"]["children"][1]["type"], "continue");
}

TEST(JSONSerializerTest, ReturnToJSON) {
    // Create a sample Return node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& scope = builder.add_return(root, "");

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Return node to JSON
    nlohmann::json j;
    serializer.return_node_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "return");
    EXPECT_TRUE(j.contains("data"));
    EXPECT_EQ(j["data"], "");
    EXPECT_FALSE(j.contains("data_type"));
}

TEST(JSONSerializerTest, SequenceToJSON) {
    // Create a sample Sequence node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& block1 = builder.add_block(scope);
    auto& block2 = builder.add_assignments(scope, {{symbolic::symbol("i"), symbolic::integer(0)}});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.sequence_to_json(j, scope);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "sequence");
    EXPECT_TRUE(j.contains("children"));
    EXPECT_EQ(j["children"].size(), 2);
    EXPECT_EQ(j["children"][0]["type"], "block");
    EXPECT_EQ(j["children"][1]["type"], "assignment_block");

    EXPECT_FALSE(j.contains("transitions"));
    EXPECT_FALSE(j["children"][0].contains("assignments"));

    EXPECT_TRUE(j["children"][1].contains("assignments"));
    EXPECT_TRUE(j["children"][1]["assignments"].is_array());
    EXPECT_EQ(j["children"][1]["assignments"].size(), 1);
    EXPECT_TRUE(j["children"][1]["assignments"][0].contains("symbol"));
    EXPECT_EQ(j["children"][1]["assignments"][0]["symbol"], "i");
    EXPECT_TRUE(j["children"][1]["assignments"][0].contains("expression"));
    // EXPECT_TRUE(j["children"][1]["assignments"][0]["expression"], "0");
}

TEST(JSONSerializerTest, MapToJSON) {
    // Create a sample Map node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);

    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = builder.add_block(map.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Map node to JSON
    nlohmann::json j;
    serializer.structured_loop_to_json(j, map);

    // Check if the JSON contains the expected keys
    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "map");
    EXPECT_TRUE(j.contains("sub_type"));
    EXPECT_EQ(j["sub_type"], "map");
    EXPECT_TRUE(j.contains("indvar"));
    EXPECT_EQ(j["indvar"], "i");
    EXPECT_TRUE(j.contains("init"));
    EXPECT_EQ(j["init"], "0");
    EXPECT_TRUE(j.contains("update"));
    EXPECT_TRUE(symbolic::
                    eq(SymEngine::Expression(j["update"]), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));
    EXPECT_TRUE(j.contains("condition"));
    EXPECT_EQ(j["condition"], "(i < 10)");
    EXPECT_TRUE(j.contains("root"));
    EXPECT_EQ(j["root"]["type"], "sequence");
    EXPECT_EQ(j["root"]["children"].size(), 1);
}

TEST(JSONSerializerTest, ReduceToJSON) {
    // Create a sample Reduce node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& reduce = builder.add_reduce(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Add, "i"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& body = builder.add_block(reduce.root());

    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    nlohmann::json j;
    serializer.structured_loop_to_json(j, reduce);

    EXPECT_TRUE(j.contains("type"));
    EXPECT_EQ(j["type"], "for");
    EXPECT_TRUE(j.contains("sub_type"));
    EXPECT_EQ(j["sub_type"], "reduce");
    EXPECT_TRUE(j.contains("indvar"));
    EXPECT_EQ(j["indvar"], "i");
    EXPECT_TRUE(j.contains("init"));
    EXPECT_EQ(j["init"], "0");
    EXPECT_TRUE(j.contains("update"));
    EXPECT_TRUE(symbolic::
                    eq(SymEngine::Expression(j["update"]), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));
    EXPECT_TRUE(j.contains("condition"));
    EXPECT_EQ(j["condition"], "(i < 10)");
    EXPECT_TRUE(j.contains("reductions"));
    EXPECT_TRUE(j["reductions"].is_array());
    EXPECT_EQ(j["reductions"].size(), 1);
    EXPECT_EQ(j["reductions"][0]["op"], "add");
    EXPECT_EQ(j["reductions"][0]["container"], "i");
    EXPECT_TRUE(j.contains("schedule_type"));
    EXPECT_TRUE(j.contains("root"));
    EXPECT_EQ(j["root"]["type"], "sequence");
    EXPECT_EQ(j["root"]["children"].size(), 1);
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Scalar) {
    // Create a sample Scalar data type
    types::Scalar scalar_type(types::StorageType::CPU_Stack(), 0, "initializer", types::PrimitiveType::Int32);
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, scalar_type);

    // Deserialize the JSON back into a Scalar data type
    auto deserialized_scalar_type = serializer.json_to_type(j);
    EXPECT_TRUE(deserialized_scalar_type != nullptr);
    EXPECT_TRUE(dynamic_cast<types::Scalar*>(deserialized_scalar_type.get()) != nullptr);
    auto scalar_ptr = dynamic_cast<types::Scalar*>(deserialized_scalar_type.get());
    EXPECT_TRUE(scalar_ptr != nullptr);

    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(scalar_ptr->primitive_type(), scalar_type.primitive_type());
    EXPECT_EQ(scalar_ptr->storage_type(), scalar_type.storage_type());
    EXPECT_EQ(scalar_ptr->alignment(), scalar_type.alignment());
    EXPECT_EQ(scalar_ptr->initializer(), scalar_type.initializer());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Pointer) {
    // Create a sample Pointer data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(types::StorageType::CPU_Stack(), 0, "initializer", base_desc);
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, pointer_type);

    // Deserialize the JSON back into a Pointer data type
    auto deserialized_pointer_type = serializer.json_to_type(j);

    // Check if the deserialized data type is of type Pointer
    EXPECT_TRUE(dynamic_cast<types::Pointer*>(deserialized_pointer_type.get()) != nullptr);
    auto pointer_ptr = dynamic_cast<types::Pointer*>(deserialized_pointer_type.get());

    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(pointer_ptr->storage_type().value(), pointer_type.storage_type().value());
    EXPECT_EQ(pointer_ptr->alignment(), pointer_type.alignment());
    EXPECT_EQ(pointer_ptr->initializer(), pointer_type.initializer());
    EXPECT_EQ(pointer_ptr->pointee_type().primitive_type(), base_desc.primitive_type());

    // Check if the deserialized pointee data type matches the original pointee data type
    auto deserialized_pointee_type = &pointer_ptr->pointee_type();
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(deserialized_pointee_type) != nullptr);
    auto deserialized_base_desc = dynamic_cast<const types::Scalar*>(deserialized_pointee_type);
    EXPECT_TRUE(deserialized_base_desc != nullptr);
    EXPECT_EQ(deserialized_base_desc->primitive_type(), base_desc.primitive_type());
    EXPECT_EQ(deserialized_base_desc->storage_type().value(), base_desc.storage_type().value());
    EXPECT_EQ(deserialized_base_desc->alignment(), base_desc.alignment());
    EXPECT_EQ(deserialized_base_desc->initializer(), base_desc.initializer());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Structure) {
    // Create a sample Structure data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Structure structure_type(types::StorageType::CPU_Stack(), 0, "initializer", "MyStruct");
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();

    sdfg::serializer::JSONSerializer serializer;
    // Serialize the data type to JSON
    serializer.type_to_json(j, structure_type);
    // Deserialize the JSON back into a Structure data type
    auto deserialized_structure_type = serializer.json_to_type(j);

    // Check if the deserialized data type is of type Structure
    EXPECT_TRUE(dynamic_cast<types::Structure*>(deserialized_structure_type.get()) != nullptr);
    auto structure_ptr = dynamic_cast<types::Structure*>(deserialized_structure_type.get());
    EXPECT_TRUE(structure_ptr != nullptr);
    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(structure_ptr->name(), structure_type.name());
    EXPECT_EQ(structure_ptr->storage_type().value(), structure_type.storage_type().value());
    EXPECT_EQ(structure_ptr->alignment(), structure_type.alignment());
    EXPECT_EQ(structure_ptr->initializer(), structure_type.initializer());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Array) {
    // Create a sample Array data type
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_type(types::StorageType::CPU_Stack(), 0, "initializer", base_desc, {symbolic::symbol("N")});
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, array_type);

    // Deserialize the JSON back into an Array data type
    auto deserialized_array_type = serializer.json_to_type(j);

    // Check if the deserialized data type is of type Array
    EXPECT_TRUE(dynamic_cast<types::Array*>(deserialized_array_type.get()) != nullptr);
    auto array_ptr = dynamic_cast<types::Array*>(deserialized_array_type.get());
    EXPECT_TRUE(array_ptr != nullptr);

    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(array_ptr->storage_type().value(), array_type.storage_type().value());
    EXPECT_EQ(array_ptr->alignment(), array_type.alignment());
    EXPECT_EQ(array_ptr->initializer(), array_type.initializer());
    EXPECT_EQ(array_ptr->num_elements()->__str__(), "N");

    // Check if the deserialized element type matches the original element type
    auto deserialized_element_type = &array_ptr->element_type();
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(deserialized_element_type) != nullptr);
    auto deserialized_base_desc = dynamic_cast<const types::Scalar*>(deserialized_element_type);
    EXPECT_TRUE(deserialized_base_desc != nullptr);
    EXPECT_EQ(deserialized_base_desc->primitive_type(), base_desc.primitive_type());
    EXPECT_EQ(deserialized_base_desc->storage_type().value(), base_desc.storage_type().value());
    EXPECT_EQ(deserialized_base_desc->alignment(), base_desc.alignment());
    EXPECT_EQ(deserialized_base_desc->initializer(), base_desc.initializer());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_Function) {
    // Create a sample Scalar data type
    types::Scalar scalar_type(types::StorageType::CPU_Stack(), 0, "initializer", types::PrimitiveType::Int32);
    types::Function function_type(scalar_type, false);
    function_type.add_param(scalar_type);
    function_type.add_param(scalar_type);
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the data type to JSON
    serializer.type_to_json(j, function_type);

    // Deserialize the JSON back into a Scalar data type
    auto deserialized_function_type = serializer.json_to_type(j);
    EXPECT_TRUE(deserialized_function_type != nullptr);
    EXPECT_TRUE(dynamic_cast<types::Function*>(deserialized_function_type.get()) != nullptr);
    auto function_ptr = dynamic_cast<types::Function*>(deserialized_function_type.get());
    EXPECT_TRUE(function_ptr != nullptr);

    // Check if the deserialized data type matches the original data type
    EXPECT_EQ(function_ptr->primitive_type(), function_type.primitive_type());
    EXPECT_EQ(function_ptr->storage_type().value(), function_type.storage_type().value());
    EXPECT_EQ(function_ptr->alignment(), function_type.alignment());
    EXPECT_EQ(function_ptr->initializer(), function_type.initializer());
    EXPECT_EQ(function_ptr->is_var_arg(), function_type.is_var_arg());
    EXPECT_EQ(function_ptr->num_params(), function_type.num_params());
    EXPECT_EQ(function_ptr->num_params(), 2);
    EXPECT_EQ(function_ptr->param_type(symbolic::integer(0)).primitive_type(), scalar_type.primitive_type());
    EXPECT_EQ(function_ptr->param_type(symbolic::integer(1)).primitive_type(), scalar_type.primitive_type());
}

TEST(JSONSerializerTest, SerializeDeserializeDataType_StructureDefinition) {
    // Create a sample StructureDefinition object
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Array array_type(base_desc, {symbolic::symbol("N")});
    types::StructureDefinition structure_definition("MyStruct", true);
    structure_definition.add_member(base_desc);
    structure_definition.add_member(array_type);
    nlohmann::json j;

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the StructureDefinition to JSON
    serializer.structure_definition_to_json(j, structure_definition);

    // define sdfg builder for deserialization
    sdfg::builder::StructuredSDFGBuilder builder_deserialize("test_sdfg", FunctionType_CPU);

    // Deserialize the JSON back into a StructureDefinition object
    serializer.json_to_structure_definition(j, builder_deserialize);

    auto des_sdfg = builder_deserialize.move();

    EXPECT_EQ(des_sdfg->structures().size(), 1);
    auto& deserialized_structure_definition = des_sdfg->structures().begin().base()->second;

    // Check if the deserialized StructureDefinition matches the original StructureDefinition
    EXPECT_EQ(deserialized_structure_definition->name(), structure_definition.name());
    EXPECT_EQ(deserialized_structure_definition->num_members(), structure_definition.num_members());
    EXPECT_EQ(deserialized_structure_definition->is_packed(), structure_definition.is_packed());

    auto& des_member_0 = deserialized_structure_definition->member_type(symbolic::integer(0));
    auto& member_0 = structure_definition.member_type(symbolic::integer(0));
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&des_member_0) != nullptr);
    EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&member_0) != nullptr);
    EXPECT_EQ(des_member_0.primitive_type(), member_0.primitive_type());
    EXPECT_EQ(des_member_0.storage_type().value(), member_0.storage_type().value());
    EXPECT_EQ(des_member_0.initializer(), member_0.initializer());
    EXPECT_EQ(des_member_0.alignment(), member_0.alignment());

    auto& des_member_1 = deserialized_structure_definition->member_type(symbolic::integer(1));
    auto& member_1 = structure_definition.member_type(symbolic::integer(1));
    EXPECT_TRUE(dynamic_cast<const types::Array*>(&des_member_1) != nullptr);
    auto& des_member_1_arr = dynamic_cast<const types::Array&>(des_member_1);
    EXPECT_TRUE(dynamic_cast<const types::Array*>(&member_1) != nullptr);
    auto& member_1_arr = dynamic_cast<const types::Array&>(member_1);

    EXPECT_EQ(des_member_1_arr.element_type().primitive_type(), member_1_arr.element_type().primitive_type());
    EXPECT_EQ(des_member_1_arr.storage_type().value(), member_1_arr.storage_type().value());
    EXPECT_EQ(des_member_1_arr.initializer(), member_1_arr.initializer());
    EXPECT_EQ(des_member_1_arr.alignment(), member_1_arr.alignment());
    EXPECT_TRUE(sdfg::symbolic::eq(des_member_1_arr.num_elements(), member_1_arr.num_elements()));
}

TEST(JSONSerializerTest, SerializeDeserialize_DataflowGraph) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc);
    builder.add_container("C", base_desc);

    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    auto& memlet_in =
        builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    auto& memlet_in2 = builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    // Serialize the DataflowGraph to JSON
    auto& block_new = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(sdfg->root().at(0));

    serializer.dataflow_to_json(j, block_new.dataflow());

    // Deserialize the JSON back into a DataflowGraph object
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    auto& block2 = des_builder.add_block(des_builder.subject().root());

    des_builder.add_container("A", opaque_desc);
    des_builder.add_container("C", base_desc);

    serializer.json_to_dataflow(j, des_builder, block2);
    auto des_sdfg = des_builder.move();

    auto& des_block_new = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(des_sdfg->root().at(0));

    auto& des_dataflow = des_block_new.dataflow();
    auto& data_flow = block_new.dataflow();
    EXPECT_EQ(des_dataflow.nodes().size(), data_flow.nodes().size());
    EXPECT_EQ(des_dataflow.edges().size(), data_flow.edges().size());

    bool foundA = false;
    int foundC = 0;
    bool found_tasklet = false;

    // Check if the deserialized DataflowGraph matches the original DataflowGraph
    for (const auto& node : des_dataflow.nodes()) {
        if (auto access_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                foundA = true;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Pointer*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Pointer&>(type);
                EXPECT_EQ(type_ptr.storage_type().value(), opaque_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), opaque_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), opaque_desc.alignment());
                EXPECT_FALSE(type_ptr.has_pointee_type());
            } else if (access_node->data() == "C") {
                foundC++;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Scalar&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), base_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
            }

        } else if (auto tasklet_node = dynamic_cast<const sdfg::data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::fp_add);
            EXPECT_EQ(tasklet_node->inputs().size(), 2);
            EXPECT_EQ(tasklet_node->output(), "_out");
            EXPECT_EQ(tasklet_node->inputs().at(0), "_in1");
            EXPECT_EQ(tasklet_node->inputs().at(1), "_in2");
            found_tasklet = true;
        }
    }

    EXPECT_TRUE(foundA);
    EXPECT_EQ(foundC, 2);
    EXPECT_TRUE(found_tasklet);
    bool found_memlet_in = false;
    bool found_memlet_in2 = false;
    bool found_memlet_out = false;
    // Check if the deserialized Memlets match the original Memlets
    for (const auto& edge : des_dataflow.edges()) {
        if (auto memlet = dynamic_cast<const sdfg::data_flow::Memlet*>(&edge)) {
            if (memlet->dst_conn() == "_in1") {
                found_memlet_in = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "A");

                auto& subset = memlet->subset();
                EXPECT_EQ(subset.size(), 1);
                EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("i")));
            } else if (memlet->dst_conn() == "_in2") {
                found_memlet_in2 = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "C");
            } else if (memlet->src_conn() == "_out") {
                found_memlet_out = true;
                auto& dst = memlet->dst();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&dst) != nullptr);
                auto& dst_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(dst);
                EXPECT_EQ(dst_ptr.data(), "C");
            }
        }
    }

    EXPECT_TRUE(found_memlet_in);
    EXPECT_TRUE(found_memlet_in2);
    EXPECT_TRUE(found_memlet_out);
}

TEST(JSONSerializerTest, SerializeDeserializeBlock_DataflowGraph) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc);
    builder.add_container("C", base_desc);

    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    auto& memlet_in =
        builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    auto& memlet_in2 = builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    // Serialize the DataflowGraph to JSON
    auto& block_new = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(sdfg->root().at(0));

    serializer.block_to_json(j, block_new);

    // Deserialize the JSON back into a DataflowGraph object
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    des_builder.add_container("A", opaque_desc);
    des_builder.add_container("C", base_desc);

    serializer.json_to_block_node(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();

    auto& des_block_new = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(des_sdfg->root().at(0));

    auto& des_dataflow = des_block_new.dataflow();
    auto& data_flow = block_new.dataflow();
    EXPECT_EQ(des_dataflow.nodes().size(), data_flow.nodes().size());
    EXPECT_EQ(des_dataflow.edges().size(), data_flow.edges().size());

    bool foundA = false;
    int foundC = 0;
    bool found_tasklet = false;

    // Check if the deserialized DataflowGraph matches the original DataflowGraph
    for (const auto& node : des_dataflow.nodes()) {
        if (auto access_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                foundA = true;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Pointer*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Pointer&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), opaque_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), opaque_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), opaque_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), opaque_desc.alignment());
                EXPECT_FALSE(type_ptr.has_pointee_type());
            } else if (access_node->data() == "C") {
                foundC++;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Scalar&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), base_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
            }

        } else if (auto tasklet_node = dynamic_cast<const sdfg::data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::fp_add);
            EXPECT_EQ(tasklet_node->inputs().size(), 2);
            EXPECT_EQ(tasklet_node->output(), "_out");
            EXPECT_EQ(tasklet_node->inputs().at(0), "_in1");
            EXPECT_EQ(tasklet_node->inputs().at(1), "_in2");
            found_tasklet = true;
        }
    }

    EXPECT_TRUE(foundA);
    EXPECT_EQ(foundC, 2);
    EXPECT_TRUE(found_tasklet);
    bool found_memlet_in = false;
    bool found_memlet_in2 = false;
    bool found_memlet_out = false;
    // Check if the deserialized Memlets match the original Memlets
    for (const auto& edge : des_dataflow.edges()) {
        if (auto memlet = dynamic_cast<const sdfg::data_flow::Memlet*>(&edge)) {
            if (memlet->dst_conn() == "_in1") {
                found_memlet_in = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "A");

                auto& subset = memlet->subset();
                EXPECT_EQ(subset.size(), 1);
                EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("i")));
            } else if (memlet->dst_conn() == "_in2") {
                found_memlet_in2 = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "C");
            } else if (memlet->src_conn() == "_out") {
                found_memlet_out = true;
                auto& dst = memlet->dst();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&dst) != nullptr);
                auto& dst_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(dst);
                EXPECT_EQ(dst_ptr.data(), "C");
            }
        }
    }

    EXPECT_TRUE(found_memlet_in);
    EXPECT_TRUE(found_memlet_in2);
    EXPECT_TRUE(found_memlet_out);
}

TEST(JSONSerializerTest, SerializeDeserializeSequence_DataflowGraph) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc);
    builder.add_container("C", base_desc);

    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    auto& memlet_in =
        builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    auto& memlet_in2 = builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    // Serialize the DataflowGraph to JSON
    auto& block_new = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(sdfg->root().at(0));

    serializer.sequence_to_json(j, sdfg->root());

    // Deserialize the JSON back into a DataflowGraph object
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    des_builder.add_container("A", opaque_desc);
    des_builder.add_container("C", base_desc);

    serializer.json_to_sequence(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();

    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 2);
    EXPECT_EQ(des_sdfg->root().size(), 1);

    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Block*>(&des_sdfg->root().at(0)) != nullptr);

    auto& des_block_new = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(des_sdfg->root().at(0));

    auto& des_dataflow = des_block_new.dataflow();
    auto& data_flow = block_new.dataflow();
    EXPECT_EQ(des_dataflow.nodes().size(), data_flow.nodes().size());
    EXPECT_EQ(des_dataflow.edges().size(), data_flow.edges().size());

    bool foundA = false;
    int foundC = 0;
    bool found_tasklet = false;

    // Check if the deserialized DataflowGraph matches the original DataflowGraph
    for (const auto& node : des_dataflow.nodes()) {
        if (auto access_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                foundA = true;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Pointer*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Pointer&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), opaque_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), opaque_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), opaque_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), opaque_desc.alignment());
                EXPECT_FALSE(type_ptr.has_pointee_type());
            } else if (access_node->data() == "C") {
                foundC++;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Scalar&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), base_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
            }

        } else if (auto tasklet_node = dynamic_cast<const sdfg::data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::fp_add);
            EXPECT_EQ(tasklet_node->inputs().size(), 2);
            EXPECT_EQ(tasklet_node->output(), "_out");
            EXPECT_EQ(tasklet_node->inputs().at(0), "_in1");
            EXPECT_EQ(tasklet_node->inputs().at(1), "_in2");
            found_tasklet = true;
        }
    }

    EXPECT_TRUE(foundA);
    EXPECT_EQ(foundC, 2);
    EXPECT_TRUE(found_tasklet);
    bool found_memlet_in = false;
    bool found_memlet_in2 = false;
    bool found_memlet_out = false;
    // Check if the deserialized Memlets match the original Memlets
    for (const auto& edge : des_dataflow.edges()) {
        if (auto memlet = dynamic_cast<const sdfg::data_flow::Memlet*>(&edge)) {
            if (memlet->dst_conn() == "_in1") {
                found_memlet_in = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "A");

                auto& subset = memlet->subset();
                EXPECT_EQ(subset.size(), 1);
                EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("i")));
            } else if (memlet->dst_conn() == "_in2") {
                found_memlet_in2 = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "C");
            } else if (memlet->src_conn() == "_out") {
                found_memlet_out = true;
                auto& dst = memlet->dst();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&dst) != nullptr);
                auto& dst_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(dst);
                EXPECT_EQ(dst_ptr.data(), "C");
            }
        }
    }

    EXPECT_TRUE(found_memlet_in);
    EXPECT_TRUE(found_memlet_in2);
    EXPECT_TRUE(found_memlet_out);
}

TEST(JSONSerializerTest, SerializeDeserializeSDFG_DataflowGraph) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(base_desc);

    types::Pointer opaque_desc;
    builder.add_container("A", opaque_desc);
    builder.add_container("C", base_desc);

    auto& block = builder.add_block(root);
    auto& access_in = builder.add_access(block, "A");
    auto& access_in2 = builder.add_access(block, "C");
    auto& access_out = builder.add_access(block, "C");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    auto& memlet_in =
        builder.add_computational_memlet(block, access_in, tasklet, "_in1", {symbolic::symbol("i")}, pointer_type);
    auto& memlet_in2 = builder.add_computational_memlet(block, access_in2, tasklet, "_in2", {});
    auto& memlet_out = builder.add_computational_memlet(block, tasklet, "_out", access_out, {});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;
    // Serialize the DataflowGraph to JSON
    auto& block_new = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(sdfg->root().at(0));

    j = serializer.serialize(*sdfg);

    // Deserialize the JSON back into a DataflowGraph object
    auto des_sdfg = serializer.deserialize(j);

    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 2);
    EXPECT_EQ(des_sdfg->root().size(), 1);

    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Block*>(&des_sdfg->root().at(0)) != nullptr);

    auto& des_block_new = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(des_sdfg->root().at(0));

    auto& des_dataflow = des_block_new.dataflow();
    auto& data_flow = block_new.dataflow();
    EXPECT_EQ(des_dataflow.nodes().size(), data_flow.nodes().size());
    EXPECT_EQ(des_dataflow.edges().size(), data_flow.edges().size());

    bool foundA = false;
    int foundC = 0;
    bool found_tasklet = false;

    // Check if the deserialized DataflowGraph matches the original DataflowGraph
    for (const auto& node : des_dataflow.nodes()) {
        if (auto access_node = dynamic_cast<const sdfg::data_flow::AccessNode*>(&node)) {
            if (access_node->data() == "A") {
                foundA = true;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Pointer*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Pointer&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), opaque_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type().value(), opaque_desc.storage_type().value());
                EXPECT_EQ(type_ptr.initializer(), opaque_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), opaque_desc.alignment());
                EXPECT_FALSE(type_ptr.has_pointee_type());
            } else if (access_node->data() == "C") {
                foundC++;
                auto& type = des_sdfg->type(access_node->data());
                EXPECT_TRUE(dynamic_cast<const types::Scalar*>(&type) != nullptr);
                auto& type_ptr = dynamic_cast<const types::Scalar&>(type);
                EXPECT_EQ(type_ptr.primitive_type(), base_desc.primitive_type());
                EXPECT_EQ(type_ptr.storage_type(), base_desc.storage_type());
                EXPECT_EQ(type_ptr.initializer(), base_desc.initializer());
                EXPECT_EQ(type_ptr.alignment(), base_desc.alignment());
            }

        } else if (auto tasklet_node = dynamic_cast<const sdfg::data_flow::Tasklet*>(&node)) {
            EXPECT_EQ(tasklet_node->code(), data_flow::TaskletCode::fp_add);
            EXPECT_EQ(tasklet_node->inputs().size(), 2);
            EXPECT_EQ(tasklet_node->inputs().at(0), "_in1");
            EXPECT_EQ(tasklet_node->inputs().at(1), "_in2");
            EXPECT_EQ(tasklet_node->output(), "_out");
            found_tasklet = true;
        }
    }

    EXPECT_TRUE(foundA);
    EXPECT_EQ(foundC, 2);
    EXPECT_TRUE(found_tasklet);
    bool found_memlet_in = false;
    bool found_memlet_in2 = false;
    bool found_memlet_out = false;
    // Check if the deserialized Memlets match the original Memlets
    for (const auto& edge : des_dataflow.edges()) {
        if (auto memlet = dynamic_cast<const sdfg::data_flow::Memlet*>(&edge)) {
            if (memlet->dst_conn() == "_in1") {
                found_memlet_in = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "A");

                auto& subset = memlet->subset();
                EXPECT_EQ(subset.size(), 1);
                EXPECT_TRUE(symbolic::eq(subset[0], symbolic::symbol("i")));

                auto& base_type = memlet->base_type();
                EXPECT_EQ(base_type, pointer_type);
            } else if (memlet->dst_conn() == "_in2") {
                found_memlet_in2 = true;
                auto& src = memlet->src();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&src) != nullptr);
                auto& src_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(src);
                EXPECT_EQ(src_ptr.data(), "C");
            } else if (memlet->src_conn() == "_out") {
                found_memlet_out = true;
                auto& dst = memlet->dst();
                EXPECT_TRUE(dynamic_cast<const sdfg::data_flow::AccessNode*>(&dst) != nullptr);
                auto& dst_ptr = dynamic_cast<const sdfg::data_flow::AccessNode&>(dst);
                EXPECT_EQ(dst_ptr.data(), "C");
            }
        }
    }

    EXPECT_TRUE(found_memlet_in);
    EXPECT_TRUE(found_memlet_in2);
    EXPECT_TRUE(found_memlet_out);
}

TEST(JSONSerializerTest, SerializeDeserialize_forloop) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Int32);

    builder.add_container("i", base_desc);
    builder.add_container("N", base_desc);

    auto loopvar = symbolic::symbol("i");
    auto bound = symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N"));
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));
    auto init = symbolic::integer(0);

    auto& for_loop = builder.add_for(root, loopvar, bound, init, update);

    auto& block = builder.add_block(for_loop.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j;

    // Serialize the DataflowGraph to JSON

    serializer.structured_loop_to_json(j, for_loop);

    // Deserialize the JSON back into a DataflowGraph object
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    des_builder.add_container("i", base_desc);
    des_builder.add_container("N", base_desc);

    serializer.json_to_structured_loop_node(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();

    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 2);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::For*>(&des_sdfg->root().at(0)) != nullptr);
    auto& des_for_loop = sdfg::dyn_cast<sdfg::structured_control_flow::For&>(des_sdfg->root().at(0));
    EXPECT_TRUE(symbolic::eq(des_for_loop.indvar(), loopvar));
    EXPECT_TRUE(symbolic::eq(des_for_loop.condition(), bound));
    EXPECT_TRUE(symbolic::eq(des_for_loop.update(), update));
    EXPECT_TRUE(symbolic::eq(des_for_loop.init(), init));

    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Block*>(&des_for_loop.root().at(0)) != nullptr);
    auto& des_block_new = sdfg::dyn_cast<sdfg::structured_control_flow::Block&>(des_for_loop.root().at(0));
}

TEST(JSONSerializerTest, SerializeDeserialize_ifelse) {
    // Create a sample IfElse node
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& if_else = builder.add_if_else(root);
    auto& true_case = builder.add_case(if_else, symbolic::__true__());
    auto& false_case = builder.add_case(if_else, symbolic::__false__());
    auto& true_block = builder.add_block(true_case);
    auto& false_block = builder.add_block(false_case);

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the IfElse node to JSON
    nlohmann::json j;
    serializer.if_else_to_json(j, if_else);

    // Deserialize the JSON back into an IfElse node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);
    serializer.json_to_if_else_node(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();

    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::IfElse*>(&des_sdfg->root().at(0)) != nullptr);
    auto& des_if_else = sdfg::dyn_cast<sdfg::structured_control_flow::IfElse&>(des_sdfg->root().at(0));

    EXPECT_EQ(des_if_else.size(), 2);

    EXPECT_TRUE(symbolic::eq(des_if_else.at(0).second, symbolic::__true__()));
    EXPECT_TRUE(symbolic::eq(des_if_else.at(1).second, symbolic::__false__()));
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Sequence*>(&des_if_else.at(0).first) != nullptr);
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Sequence*>(&des_if_else.at(1).first) != nullptr);
    auto& des_true_case = sdfg::dyn_cast<sdfg::structured_control_flow::Sequence&>(des_if_else.at(0).first);
    auto& des_false_case = sdfg::dyn_cast<sdfg::structured_control_flow::Sequence&>(des_if_else.at(1).first);
    EXPECT_EQ(des_true_case.size(), 1);
    EXPECT_EQ(des_false_case.size(), 1);
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Block*>(&des_true_case.at(0)) != nullptr);
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Block*>(&des_false_case.at(0)) != nullptr);
}

TEST(JSONSerializerTest, SerializeDeserialize_sequence) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& block1 = builder.add_block(scope);
    auto& block2 = builder.add_assignments(scope, {{symbolic::symbol("i"), symbolic::integer(0)}});

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.sequence_to_json(j, scope);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);
    serializer.json_to_sequence(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 2);

    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Block*>(&des_sdfg->root().at(0)) != nullptr);
    auto* assignments = sdfg::dyn_cast<sdfg::structured_control_flow::AssignmentBlock*>(&des_sdfg->root().at(1));
    EXPECT_TRUE(assignments != nullptr);
    auto& assignment = assignments->assignments();
    EXPECT_EQ(assignment.size(), 1);
    EXPECT_TRUE(symbolic::eq(assignment.begin()->first, symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(assignment.begin()->second, symbolic::integer(0)));
}

TEST(JSONSerializerTest, SerializeDeserialize_while) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& while1 = builder.add_while(scope);
    auto& block1 = builder.add_block(while1.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, while1);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);

    serializer.json_to_while_node(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::While*>(&des_sdfg->root().at(0)) != nullptr);
    auto& des_while = sdfg::dyn_cast<sdfg::structured_control_flow::While&>(des_sdfg->root().at(0));

    EXPECT_EQ(des_while.root().size(), 1);
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Block*>(&des_while.root().at(0)) != nullptr);
}

TEST(JSONSerializerTest, SerializeDeserialize_while_break) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& while1 = builder.add_while(scope);
    auto& break1 = builder.add_break(while1.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, while1);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);

    serializer.json_to_while_node(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::While*>(&des_sdfg->root().at(0)) != nullptr);
    auto& des_while = sdfg::dyn_cast<sdfg::structured_control_flow::While&>(des_sdfg->root().at(0));
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Break*>(&des_while.root().at(0)) != nullptr);

    EXPECT_EQ(des_while.root().size(), 1);
    auto& des_break = sdfg::dyn_cast<sdfg::structured_control_flow::Break&>(des_while.root().at(0));
}

TEST(JSONSerializerTest, SerializeDeserialize_while_continue) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& scope = builder.add_sequence(root);
    auto& while1 = builder.add_while(scope);
    auto& continue1 = builder.add_continue(while1.root());

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.while_node_to_json(j, while1);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);
    des_builder.add_container("i", sym_desc);

    serializer.json_to_while_node(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 1);
    EXPECT_EQ(des_sdfg->root().size(), 1);
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::While*>(&des_sdfg->root().at(0)) != nullptr);
    auto& des_while = sdfg::dyn_cast<sdfg::structured_control_flow::While&>(des_sdfg->root().at(0));
    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Continue*>(&des_while.root().at(0)) != nullptr);
    EXPECT_EQ(des_while.root().size(), 1);
    auto& des_continue = sdfg::dyn_cast<sdfg::structured_control_flow::Continue&>(des_while.root().at(0));
}

TEST(JSONSerializerTest, SerializeDeserialize_Map) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& map = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.structured_loop_to_json(j, map);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    serializer.json_to_structured_loop_node(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 0);
    EXPECT_EQ(des_sdfg->root().size(), 1);

    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Map*>(&des_sdfg->root().at(0)) != nullptr);
    auto& des_map = sdfg::dyn_cast<sdfg::structured_control_flow::Map&>(des_sdfg->root().at(0));
    EXPECT_TRUE(symbolic::eq(des_map.indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(des_map.condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10))));
    EXPECT_TRUE(symbolic::eq(des_map.init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(des_map.update(), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));
    EXPECT_EQ(des_map.schedule_type().value(), structured_control_flow::ScheduleType_Sequential::value());

    EXPECT_EQ(des_map.root().size(), 0);
}

TEST(JSONSerializerTest, SerializeDeserialize_Reduce) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& reduce = builder.add_reduce(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        {structured_control_flow::ReductionInfo{structured_control_flow::ReductionOperation::Max, "i"}},
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    nlohmann::json j;
    serializer.structured_loop_to_json(j, reduce);

    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    serializer.json_to_structured_loop_node(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 0);
    EXPECT_EQ(des_sdfg->root().size(), 1);

    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Reduce*>(&des_sdfg->root().at(0)) != nullptr);
    auto& des_reduce = sdfg::dyn_cast<sdfg::structured_control_flow::Reduce&>(des_sdfg->root().at(0));
    EXPECT_TRUE(symbolic::eq(des_reduce.indvar(), symbolic::symbol("i")));
    EXPECT_TRUE(symbolic::eq(des_reduce.condition(), symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10))));
    EXPECT_TRUE(symbolic::eq(des_reduce.init(), symbolic::integer(0)));
    EXPECT_TRUE(symbolic::eq(des_reduce.update(), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))));
    ASSERT_EQ(des_reduce.reductions().size(), 1u);
    EXPECT_EQ(des_reduce.reductions()[0].operation, structured_control_flow::ReductionOperation::Max);
    EXPECT_EQ(des_reduce.reductions()[0].container, "i");
    EXPECT_EQ(des_reduce.schedule_type().value(), structured_control_flow::ScheduleType_Sequential::value());

    EXPECT_EQ(des_reduce.root().size(), 0);
}

TEST(JSONSerializerTest, SerializeDeserialize_return) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& ret = builder.add_return(root, "");

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    auto sdfg = builder.move();
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the Sequence node to JSON
    nlohmann::json j;
    serializer.return_node_to_json(j, ret);

    // Deserialize the JSON back into a Sequence node
    auto des_builder = sdfg::builder::StructuredSDFGBuilder("test_sdfg", FunctionType_CPU);

    serializer.json_to_return_node(j, des_builder, des_builder.subject().root());
    auto des_sdfg = des_builder.move();
    EXPECT_EQ(des_sdfg->name(), sdfg->name());
    EXPECT_EQ(des_sdfg->containers().size(), 0);
    EXPECT_EQ(des_sdfg->root().size(), 1);

    EXPECT_TRUE(sdfg::dyn_cast<sdfg::structured_control_flow::Return*>(&des_sdfg->root().at(0)) != nullptr);
    auto& des_ret = sdfg::dyn_cast<sdfg::structured_control_flow::Return&>(des_sdfg->root().at(0));
    EXPECT_EQ(des_ret.data(), "");
}

TEST(JSONSerializerTest, SerializeDeserialize) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto sdfg = builder.move();

    sdfg->add_metadata("key", "value");

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the SDFG to JSON
    auto j = serializer.serialize(*sdfg);

    // Deserialize the JSON back into a StructuredSDFG object
    auto sdfg_new = serializer.deserialize(j);

    // Check if the deserialized SDFG matches the original SDFG
    EXPECT_EQ(sdfg_new->name(), "test_sdfg");
    EXPECT_EQ(sdfg_new->metadata("key"), "value");
    EXPECT_EQ(sdfg_new->type(), FunctionType_CPU);
}

TEST(JSONSerializerTest, SerializeDeserialize_Arguments) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);

    builder.add_container("A", base_desc, true);
    builder.add_container("C", base_desc, false);
    builder.add_container("B", base_desc, true);

    auto sdfg = builder.move();

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the SDFG to JSON
    auto j = serializer.serialize(*sdfg);

    // Deserialize the JSON back into a StructuredSDFG object
    auto sdfg_new = serializer.deserialize(j);

    // Check if the deserialized SDFG matches the original SDFG
    EXPECT_EQ(sdfg_new->name(), "test_sdfg");
    EXPECT_EQ(sdfg_new->containers().size(), 3);
    EXPECT_EQ(sdfg_new->type(), FunctionType_CPU);

    EXPECT_EQ(sdfg_new->arguments().size(), 2);
    EXPECT_EQ(sdfg_new->arguments().at(0), "A");
    EXPECT_EQ(sdfg_new->arguments().at(1), "B");

    EXPECT_EQ(sdfg_new->root().size(), 0);
}

TEST(JSONSerializerTest, SerializeDeserialize_Externals) {
    // Create a sample StructuredSDFG object
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);

    // Add containers
    types::Scalar base_desc(types::PrimitiveType::Float);

    builder.add_external("A", base_desc, LinkageType_External);
    builder.add_external("C", base_desc, LinkageType_Internal);
    builder.add_external("B", base_desc, LinkageType_External);

    auto sdfg = builder.move();

    // Create a JSONSerializer object
    std::string filename = "test_sdfg.json";
    sdfg::serializer::JSONSerializer serializer;

    // Serialize the SDFG to JSON
    auto j = serializer.serialize(*sdfg);

    // Deserialize the JSON back into a StructuredSDFG object
    auto sdfg_new = serializer.deserialize(j);

    // Check if the deserialized SDFG matches the original SDFG
    EXPECT_EQ(sdfg_new->name(), "test_sdfg");
    EXPECT_EQ(sdfg_new->containers().size(), 3);
    EXPECT_EQ(sdfg_new->type(), FunctionType_CPU);

    EXPECT_EQ(sdfg_new->externals().size(), 3);
    EXPECT_EQ(sdfg_new->linkage_type("A"), LinkageType_External);
    EXPECT_EQ(sdfg_new->linkage_type("B"), LinkageType_External);
    EXPECT_EQ(sdfg_new->linkage_type("C"), LinkageType_Internal);

    EXPECT_EQ(sdfg_new->root().size(), 0);
}

TEST(JSONSerializerTest, SerializeDeserialize_LibraryNode) {
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& lib_node = builder.add_library_node<data_flow::BarrierLocalNode>(block, DebugInfo());

    // Get the library node serializer
    auto lib_node_serializer_fn = serializer::LibraryNodeSerializerRegistry::instance()
                                      .get_library_node_serializer(data_flow::LibraryNodeType_BarrierLocal.value());
    EXPECT_TRUE(lib_node_serializer_fn != nullptr);
    auto lib_node_serializer_ptr = lib_node_serializer_fn();
    EXPECT_TRUE(lib_node_serializer_ptr != nullptr);

    // Serialize the library node
    auto j = lib_node_serializer_ptr->serialize(lib_node);

    // Deserialize the library node
    auto& lib_node_new = lib_node_serializer_ptr->deserialize(j, builder, block);

    EXPECT_EQ(lib_node_new.code(), data_flow::LibraryNodeType_BarrierLocal);
    EXPECT_EQ(lib_node_new.side_effect(), lib_node.side_effect());
    EXPECT_EQ(lib_node_new.outputs(), lib_node.outputs());
    EXPECT_EQ(lib_node_new.inputs(), lib_node.inputs());

    EXPECT_TRUE(dynamic_cast<data_flow::BarrierLocalNode*>(&lib_node_new));
    auto barrier_local_node = dynamic_cast<data_flow::BarrierLocalNode*>(&lib_node_new);
}

TEST(JsonSerializerTest, DeseríalizeLegacyTransitions) {
    std::string json_str = R"a(
{
    "children": [
        {
            "dataflow": {
                "edges": [],
                "nodes": [],
                "type": "dataflow"
            },
            "debug_info": {
                "end_column": 38,
                "end_line": 39,
                "filename": "",
                "function": "init_array",
                "has": true,
                "start_column": 38,
                "start_line": 39
            },
            "element_id": 69,
            "type": "block"
        },
        {
            "dataflow": {
                "edges": [],
                "nodes": [],
                "type": "dataflow"
            },
            "debug_info": {
                "end_column": 38,
                "end_line": 39,
                "filename": "",
                "function": "init_array",
                "has": true,
                "start_column": 38,
                "start_line": 39
            },
            "element_id": 69,
            "type": "block"
        }
    ],
    "debug_info": {
        "end_column": 15,
        "end_line": 39,
        "filename": "",
        "function": "",
        "has": true,
        "start_column": 19,
        "start_line": 38
    },
    "element_id": 1392,
    "transitions": [
        {
            "assignments": [
                {
                    "expression": "1 + trunc_i32(_7*_12)",
                    "symbol": "_17"
                }
            ],
            "debug_info": {
                "end_column": 38,
                "end_line": 39,
                "filename": "",
                "function": "init_array",
                "has": true,
                "start_column": 38,
                "start_line": 39
            },
            "element_id": 70,
            "type": "transition"
        },
        {
            "assignments": [],
            "debug_info": {
                "end_column": 38,
                "end_line": 39,
                "filename": "",
                "function": "init_array",
                "has": true,
                "start_column": 38,
                "start_line": 39
            },
            "element_id": 77,
            "type": "transition"
        }
    ],
    "type": "sequence"
}
)a";

    nlohmann::json j = nlohmann::json::parse(json_str);

    serializer::JSONSerializer ser;
    builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    ser.json_to_sequence(j, builder, builder.subject().root());

    auto& root_seq = builder.subject().root();
    EXPECT_EQ(root_seq.size(), 4);

    EXPECT_TRUE(dyn_cast<Block*>(&root_seq.at(0)));
    EXPECT_TRUE(dyn_cast<Block*>(&root_seq.at(2)));
    auto& recovered_transition1 = root_seq.at(1);
    auto& recovered_transition2 = root_seq.at(3);
    auto assignBlock1 = dyn_cast<AssignmentBlock*>(&recovered_transition1);
    EXPECT_TRUE(assignBlock1);
    EXPECT_EQ(assignBlock1->element_id(), 70);
    EXPECT_EQ(assignBlock1->size(), 1);
    auto s = symbolic::symbol("_17");
    auto expr = assignBlock1->assignments().at(s);
    EXPECT_TRUE(symbolic::null_safe_eq(expr, symbolic::parse("1 + trunc_i32(_7*_12)")));
    auto assignBlock2 = dyn_cast<AssignmentBlock*>(&recovered_transition2);
    EXPECT_TRUE(assignBlock2);
    EXPECT_EQ(assignBlock2->element_id(), 77);
    EXPECT_EQ(assignBlock2->size(), 0);
}
