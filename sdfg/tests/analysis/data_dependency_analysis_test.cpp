#include "sdfg/analysis/data_dependency_analysis.h"

#include <gtest/gtest.h>

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"

using namespace sdfg;

TEST(DataDependencyAnalysisTest, Block_Define_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& input_node = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& write_A = *open_definitions.begin();
    EXPECT_EQ(write_A.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_A.first->container(), "A");
    EXPECT_EQ(write_A.first->element(), &output_node);
    EXPECT_EQ(write_A.second.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Define_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& input_node = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& write_A = *open_definitions.begin();
    EXPECT_EQ(write_A.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_A.first->container(), "A");
    EXPECT_EQ(write_A.first->element(), &output_node);
    EXPECT_EQ(write_A.second.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Undefined_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);
    builder.add_container("B", base_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& write_B = *open_definitions.begin();
    EXPECT_EQ(write_B.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_B.first->container(), "B");
    EXPECT_EQ(write_B.first->element(), &output_node);
    EXPECT_EQ(write_B.second.size(), 0);

    auto& read_A = *undefined.begin();
    EXPECT_EQ(read_A->use(), analysis::Use::READ);
    EXPECT_EQ(read_A->container(), "A");
    EXPECT_EQ(read_A->element(), &input_node);
}

TEST(DataDependencyAnalysisTest, Block_Undefined_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);
    builder.add_container("B", array_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& write_B = *open_definitions.begin();
    EXPECT_EQ(write_B.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_B.first->container(), "B");
    EXPECT_EQ(write_B.first->element(), &output_node);
    EXPECT_EQ(write_B.second.size(), 0);

    auto& read_A = *undefined.begin();
    EXPECT_EQ(read_A->use(), analysis::Use::READ);
    EXPECT_EQ(read_A->container(), "A");
    EXPECT_EQ(read_A->element(), &input_node);
}

TEST(DataDependencyAnalysisTest, Block_Undefined_Array_Subset) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);
    builder.add_container("B", array_desc);
    builder.add_container("C", array_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {symbolic::integer(0)});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet2, "_out", output_node2, {symbolic::integer(0)});
    builder.add_computational_memlet(block, output_node, tasklet2, "_in", {symbolic::integer(1)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    analysis.set_detailed(true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 2);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto read_A = users.get_user("A", &input_node, analysis::Use::READ);
    auto undefined_A = undefined.find(read_A);
    EXPECT_NE(undefined_A, undefined.end());

    auto read_B = users.get_user("B", &output_node, analysis::Use::READ);
    auto undefined_B = undefined.find(read_B);
    EXPECT_NE(undefined_B, undefined.end());

    auto write_B = users.get_user("B", &output_node, analysis::Use::WRITE);
    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 0);

    auto write_C = users.get_user("C", &output_node2, analysis::Use::WRITE);
    auto& definition_C = open_definitions.at(write_C);
    EXPECT_EQ(definition_C.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Undefined_Symbol) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);
    builder.add_container("i", base_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    auto& memlet = builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::symbol("i")});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& read_i = *undefined.begin();
    EXPECT_EQ(read_i->use(), analysis::Use::READ);
    EXPECT_EQ(read_i->container(), "i");
    EXPECT_EQ(read_i->element(), &memlet);

    auto& write_A = *open_definitions.begin();
    EXPECT_EQ(write_A.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_A.first->container(), "A");
    EXPECT_EQ(write_A.first->element(), &output_node);
    EXPECT_EQ(write_A.second.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Use_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);
    builder.add_container("B", base_desc);
    builder.add_container("C", base_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet2, "_out", output_node2, {});
    builder.add_computational_memlet(block, output_node, tasklet2, "_in", {});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& read_A = *undefined.begin();
    EXPECT_EQ(read_A->use(), analysis::Use::READ);
    EXPECT_EQ(read_A->container(), "A");
    EXPECT_EQ(read_A->element(), &input_node);

    auto write_B = users.get_user("B", &output_node, analysis::Use::WRITE);
    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 1);
    EXPECT_EQ((*definition_B.begin())->use(), analysis::Use::READ);
    EXPECT_EQ((*definition_B.begin())->container(), "B");
    EXPECT_EQ((*definition_B.begin())->element(), &output_node);

    auto write_C = users.get_user("C", &output_node2, analysis::Use::WRITE);
    auto& definition_C = open_definitions.at(write_C);
    EXPECT_EQ(definition_C.size(), 0);
}

TEST(DataDependencyAnalysisTest, Block_Use_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);
    builder.add_container("B", array_desc);
    builder.add_container("C", array_desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& input_node = builder.add_access(block, "A");
    auto& output_node = builder.add_access(block, "B");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)});
    builder.add_computational_memlet(block, input_node, tasklet, "_in", {symbolic::integer(0)});

    auto& output_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, tasklet2, "_out", output_node2, {symbolic::integer(0)});
    builder.add_computational_memlet(block, output_node, tasklet2, "_in", {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_block(analysis_manager, block, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& read_A = *undefined.begin();
    EXPECT_EQ(read_A->use(), analysis::Use::READ);
    EXPECT_EQ(read_A->container(), "A");
    EXPECT_EQ(read_A->element(), &input_node);

    auto write_B = users.get_user("B", &output_node, analysis::Use::WRITE);
    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 1);
    EXPECT_EQ((*definition_B.begin())->use(), analysis::Use::READ);
    EXPECT_EQ((*definition_B.begin())->container(), "B");
    EXPECT_EQ((*definition_B.begin())->element(), &output_node);

    auto write_C = users.get_user("C", &output_node2, analysis::Use::WRITE);
    auto& definition_C = open_definitions.at(write_C);
    EXPECT_EQ(definition_C.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Define_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = root.at(0).second;

    auto& write_A = *open_definitions.begin();
    EXPECT_EQ(write_A.first->use(), analysis::Use::WRITE);
    EXPECT_EQ(write_A.first->container(), "A");
    EXPECT_EQ(write_A.first->element(), &transition1);
    EXPECT_EQ(write_A.second.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Use_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);
    builder.add_container("B", base_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& block2 = builder.add_block(root, {{symbolic::symbol("B"), symbolic::symbol("A")}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = root.at(0).second;
    auto& transition2 = root.at(1).second;

    auto write_A = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto write_B = users.get_user("B", &transition2, analysis::Use::WRITE);

    auto& definition_A = open_definitions.at(write_A);
    EXPECT_EQ(definition_A.size(), 1);
    EXPECT_EQ((*definition_A.begin())->use(), analysis::Use::READ);
    EXPECT_EQ((*definition_A.begin())->container(), "A");
    EXPECT_EQ((*definition_A.begin())->element(), &transition2);

    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Close_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& block2 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(1)}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto& transition1 = root.at(0).second;
    auto& transition2 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &transition2, analysis::Use::WRITE);

    auto& definition_A_1 = closed_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Close_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& output_node = builder.add_access(block1, "A");
    auto& zero_node = builder.add_constant(block1, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block1, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& one_node = builder.add_constant(block2, "1", base_desc);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, one_node, tasklet2, "_in", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    analysis.set_detailed(true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto& transition1 = root.at(0).second;
    auto& transition2 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_1 = closed_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);
}

TEST(DataDependencyAnalysisTest, Sequence_Define_Array_Subsets) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& output_node = builder.add_access(block1, "A");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(0)});

    auto& block2 = builder.add_block(root);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(1)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = root.at(0).second;
    auto& transition2 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_1 = open_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Condition_Undefined) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);
    auto& branch1 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("A"), symbolic::integer(0)));

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 1);
    EXPECT_EQ(open_definitions.size(), 0);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto read_A = users.get_user("A", &if_else, analysis::Use::READ);
    auto undefined_A = undefined.find(read_A);
    EXPECT_NE(undefined_A, undefined.end());
}

TEST(DataDependencyAnalysisTest, IfElse_Condition_Use) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& if_else = builder.add_if_else(root);
    auto& branch1 = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("A"), symbolic::integer(0)));

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = root.at(0).second;

    auto write_A = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto& definition_A = open_definitions.at(write_A);
    EXPECT_EQ(definition_A.size(), 1);
    EXPECT_EQ((*definition_A.begin())->use(), analysis::Use::READ);
    EXPECT_EQ((*definition_A.begin())->container(), "A");
    EXPECT_EQ((*definition_A.begin())->element(), &if_else);
}

TEST(DataDependencyAnalysisTest, IfElse_Define_Complete_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);

    auto& root = builder.subject().root();
    auto& block0 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& if_else = builder.add_if_else(root);
    auto& branch1 = builder.add_case(if_else, symbolic::__true__());
    auto& block1 = builder.add_block(branch1, {{symbolic::symbol("A"), symbolic::integer(1)}});
    auto& branch2 = builder.add_case(if_else, symbolic::__false__());
    auto& block2 = builder.add_block(branch2, {{symbolic::symbol("A"), symbolic::integer(2)}});
    auto& block3 = builder.add_block(root, {{symbolic::symbol("B"), symbolic::symbol("A")}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 3);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto& transition0 = root.at(0).second;
    auto& transition1 = branch1.at(0).second;
    auto& transition2 = branch2.at(0).second;
    auto& transition3 = root.at(2).second;

    auto write_A_0 = users.get_user("A", &transition0, analysis::Use::WRITE);
    auto write_A_1 = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &transition2, analysis::Use::WRITE);
    auto write_B = users.get_user("B", &transition3, analysis::Use::WRITE);
    auto read_A = users.get_user("A", &transition3, analysis::Use::READ);

    auto& definition_A_0 = closed_definitions.at(write_A_0);
    EXPECT_EQ(definition_A_0.size(), 0);

    auto& definition_A_1 = open_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 1);
    EXPECT_NE(definition_A_1.find(read_A), definition_A_1.end());

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 1);
    EXPECT_NE(definition_A_2.find(read_A), definition_A_2.end());

    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Define_Incomplete_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    builder.add_container("A", base_desc);
    builder.add_container("B", base_desc);

    auto& root = builder.subject().root();
    auto& block0 = builder.add_block(root, {{symbolic::symbol("A"), symbolic::integer(0)}});
    auto& if_else = builder.add_if_else(root);
    auto& branch1 = builder.add_case(if_else, symbolic::__false__());
    auto& block1 = builder.add_block(branch1, {{symbolic::symbol("A"), symbolic::integer(1)}});
    auto& block2 = builder.add_block(root, {{symbolic::symbol("B"), symbolic::symbol("A")}});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 4);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition0 = root.at(0).second;
    auto& transition1 = branch1.at(0).second;
    auto& transition3 = root.at(2).second;

    auto write_A_0 = users.get_user("A", &transition0, analysis::Use::WRITE);
    auto write_A_1 = users.get_user("A", &transition1, analysis::Use::WRITE);
    auto write_B = users.get_user("B", &transition3, analysis::Use::WRITE);
    auto read_A = users.get_user("A", &transition3, analysis::Use::READ);

    auto& definition_A_0 = open_definitions.at(write_A_0);
    EXPECT_EQ(definition_A_0.size(), 1);
    EXPECT_NE(definition_A_0.find(read_A), definition_A_0.end());

    auto& definition_A_1 = open_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 1);
    EXPECT_NE(definition_A_1.find(read_A), definition_A_1.end());

    auto& definition_B = open_definitions.at(write_B);
    EXPECT_EQ(definition_B.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Define_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto& branch1 = builder.add_case(if_else, symbolic::__true__());
    auto& block1 = builder.add_block(branch1);
    auto& output_node = builder.add_access(block1, "A");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(0)});

    auto& branch2 = builder.add_case(if_else, symbolic::__false__());
    auto& block2 = builder.add_block(branch2);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)});

    auto& block3 = builder.add_block(root);
    auto& output_node3 = builder.add_access(block3, "A");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block3, tasklet3, "_out", output_node3, {symbolic::integer(0)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    analysis.set_detailed(true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 1);
    EXPECT_EQ(closed_definitions.size(), 2);

    auto& transition1 = branch1.at(0).second;
    auto& transition2 = branch2.at(0).second;
    auto& transition3 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);
    auto write_A_3 = users.get_user("A", &output_node3, analysis::Use::WRITE);

    auto& definition_A_1 = closed_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = closed_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);

    auto& definition_A_3 = open_definitions.at(write_A_3);
    EXPECT_EQ(definition_A_3.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Define_Array_Subsets) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto& branch1 = builder.add_case(if_else, symbolic::__true__());
    auto& block1 = builder.add_block(branch1);
    auto& output_node = builder.add_access(block1, "A");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(0)});

    auto& branch2 = builder.add_case(if_else, symbolic::__false__());
    auto& block2 = builder.add_block(branch2);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)});

    auto& block3 = builder.add_block(root);
    auto& output_node3 = builder.add_access(block3, "A");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block3, tasklet3, "_out", output_node3, {symbolic::integer(1)});

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 3);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto& transition1 = branch1.at(0).second;
    auto& transition2 = branch2.at(0).second;
    auto& transition3 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);
    auto write_A_3 = users.get_user("A", &output_node3, analysis::Use::WRITE);

    auto& definition_A_1 = open_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);

    auto& definition_A_3 = open_definitions.at(write_A_3);
    EXPECT_EQ(definition_A_3.size(), 0);
}

TEST(DataDependencyAnalysisTest, IfElse_Close_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);

    auto& branch1 = builder.add_case(if_else, symbolic::__true__());
    auto& block1 = builder.add_block(branch1);
    auto& output_node = builder.add_access(block1, "A");
    auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block1, tasklet, "_out", output_node, {symbolic::integer(1)}, array_desc);

    auto& branch2 = builder.add_case(if_else, symbolic::__false__());
    auto& block2 = builder.add_block(branch2);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"1"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)}, array_desc);

    auto& block3 = builder.add_block(root);
    auto& output_node3 = builder.add_access(block3, "A");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block3, tasklet3, "_out", output_node3, {symbolic::integer(1)}, array_desc);

    // Run analysis
    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    analysis.set_detailed(true);
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto& transition1 = branch1.at(0).second;
    auto& transition2 = branch2.at(0).second;
    auto& transition3 = root.at(1).second;

    auto write_A_1 = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_2 = users.get_user("A", &output_node2, analysis::Use::WRITE);
    auto write_A_3 = users.get_user("A", &output_node3, analysis::Use::WRITE);

    auto& definition_A_1 = closed_definitions.at(write_A_1);
    EXPECT_EQ(definition_A_1.size(), 0);

    auto& definition_A_2 = open_definitions.at(write_A_2);
    EXPECT_EQ(definition_A_2.size(), 0);

    auto& definition_A_3 = open_definitions.at(write_A_3);
    EXPECT_EQ(definition_A_3.size(), 0);
}

TEST(DataDependencyAnalysisTest, For_Indvar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 2);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto init_i = users.get_user("i", &for_loop, analysis::Use::WRITE, true, false, false);
    auto update_i_write = users.get_user("i", &for_loop, analysis::Use::WRITE, false, false, true);
    auto update_i_read = users.get_user("i", &for_loop, analysis::Use::READ, false, false, true);
    auto condition_i = users.get_user("i", &for_loop, analysis::Use::READ, false, true, false);

    auto& definition_init_i = open_definitions.at(init_i);
    EXPECT_EQ(definition_init_i.size(), 2);
    EXPECT_NE(definition_init_i.find(update_i_read), definition_init_i.end());
    EXPECT_NE(definition_init_i.find(condition_i), definition_init_i.end());

    auto& definition_update_i_write = open_definitions.at(update_i_write);
    EXPECT_EQ(definition_update_i_write.size(), 2);
    EXPECT_NE(definition_update_i_write.find(update_i_read), definition_update_i_write.end());
    EXPECT_NE(definition_update_i_write.find(condition_i), definition_update_i_write.end());
}

TEST(DataDependencyAnalysisTest, For_Close_Scalar) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("A", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();

    auto& for_loop = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop.root());
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder
        .add_computational_memlet(block, tasklet, "_out", output_node, {}, types::Scalar(types::PrimitiveType::Int32));

    auto& block2 = builder.add_block(root);
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"0"});
    builder
        .add_computational_memlet(block2, tasklet2, "_out", output_node2, {}, types::Scalar(types::PrimitiveType::Int32));

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 3);
    EXPECT_EQ(closed_definitions.size(), 1);

    auto write_A_body = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_after = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_body = closed_definitions.at(write_A_body);
    EXPECT_EQ(definition_A_body.size(), 0);

    auto& definition_A_after = open_definitions.at(write_A_after);
    EXPECT_EQ(definition_A_after.size(), 0);
}

TEST(DataDependencyAnalysisTest, For_Open_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();

    auto& for_loop1 = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(2)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop1.root());
    auto& output_node = builder.add_access(block, "A");
    auto& zero_node = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::symbol("i")}, array_desc);

    auto& for_loop2 = builder.add_for(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(2)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    auto& block2 = builder.add_block(for_loop2.root());
    auto& output_node2 = builder.add_access(block2, "A");
    auto& zero_node2 = builder.add_constant(block2, "0", base_desc);
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, zero_node2, tasklet2, "_in", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 6);
    EXPECT_EQ(closed_definitions.size(), 0);
}

TEST(DataDependencyAnalysisTest, For_Open_Array_Subsets) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();

    auto& for_loop1 = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(2)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop1.root());
    auto& output_node = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::symbol("i")}, array_desc);

    auto& for_loop2 = builder.add_for(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    auto& block2 = builder.add_block(for_loop2.root());
    auto& output_node2 = builder.add_access(block2, "A");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"0"});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::symbol("j")}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 6);
    EXPECT_EQ(closed_definitions.size(), 0);

    auto write_A_body = users.get_user("A", &output_node, analysis::Use::WRITE);
    auto write_A_after = users.get_user("A", &output_node2, analysis::Use::WRITE);

    auto& definition_A_body = open_definitions.at(write_A_body);
    EXPECT_EQ(definition_A_body.size(), 0);

    auto& definition_A_after = open_definitions.at(write_A_after);
    EXPECT_EQ(definition_A_after.size(), 0);
}

TEST(DataDependencyAnalysisTest, For_Open_Array_Subsets_Trivial) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar base_desc(types::PrimitiveType::Int32);
    types::Array array_desc(base_desc, symbolic::integer(2));
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("j", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("A", array_desc);

    auto& root = builder.subject().root();

    auto& for_loop1 = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    auto& block = builder.add_block(for_loop1.root());
    auto& output_node = builder.add_access(block, "A");
    auto& zero_node = builder.add_constant(block, "0", base_desc);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, zero_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", output_node, {symbolic::integer(0)}, array_desc);

    auto& for_loop2 = builder.add_for(
        root,
        symbolic::symbol("j"),
        symbolic::Lt(symbolic::symbol("j"), symbolic::integer(1)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
    );

    auto& block2 = builder.add_block(for_loop2.root());
    auto& output_node2 = builder.add_access(block2, "A");
    auto& zero_node2 = builder.add_constant(block2, "0", base_desc);
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block2, zero_node2, tasklet2, "_in", {});
    builder.add_computational_memlet(block2, tasklet2, "_out", output_node2, {symbolic::integer(0)}, array_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis::DataDependencyAnalysis analysis(builder.subject());
    auto& users = analysis_manager.get<analysis::Users>();

    std::unordered_set<analysis::User*> undefined;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> open_definitions;
    std::unordered_map<analysis::User*, std::unordered_set<analysis::User*>> closed_definitions;
    analysis.visit_sequence(analysis_manager, root, undefined, open_definitions, closed_definitions);

    // Check result
    EXPECT_EQ(undefined.size(), 0);
    EXPECT_EQ(open_definitions.size(), 6);
    EXPECT_EQ(closed_definitions.size(), 0);
}
