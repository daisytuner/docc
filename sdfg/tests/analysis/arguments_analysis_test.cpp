#include "sdfg/analysis/arguments_analysis.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

TEST(ArgumentsAnalysisTest, Block_Arguments_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Add containers
    types::Scalar desc(types::PrimitiveType::Bool);
    builder.add_container("N", desc);
    builder.add_container("M", desc);

    // Add block
    auto& block = builder.add_block(builder.subject().root());

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, block));
    EXPECT_TRUE(analysis.arguments(analysis_manager, block).empty());
    EXPECT_TRUE(analysis.locals(analysis_manager, block).empty());

    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, block, false));
    EXPECT_TRUE(analysis.argument_sizes(analysis_manager, block, false).empty());
    EXPECT_TRUE(analysis.argument_element_sizes(analysis_manager, block, false).empty());
}

TEST(ArgumentsAnalysisTest, Block_Arguments_Scalars) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Add containers
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("arg1", int_type, true);
    builder.add_container("t1", int_type);

    // Add block
    auto& block = builder.add_block(builder.subject().root());

    auto& access_in = builder.add_access(block, "arg1");
    auto& access_out = builder.add_access(block, "t1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {});

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, block));

    auto arguments = analysis.arguments(analysis_manager, block);
    EXPECT_EQ(arguments.size(), 1);
    EXPECT_TRUE(arguments.contains("arg1"));
    EXPECT_TRUE(arguments.at("arg1").is_input);

    auto locals = analysis.locals(analysis_manager, block);
    EXPECT_EQ(locals.size(), 1);
    EXPECT_TRUE(locals.contains("t1"));

    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, block, false));
    auto arg_sizes = analysis.argument_sizes(analysis_manager, block, false);
    EXPECT_EQ(arg_sizes.size(), 1);
    EXPECT_TRUE(arg_sizes.contains("arg1"));
    EXPECT_TRUE(symbolic::eq(arg_sizes.at("arg1"), symbolic::integer(4)));
}

TEST(ArgumentsAnalysisTest, Block_Arguments_Arrays) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Add containers
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Scalar n_type(types::PrimitiveType::Int32);
    types::Array array_type(float_type, {symbolic::symbol("N")});
    builder.add_container("arg1", array_type, true);
    builder.add_container("t1", array_type);
    builder.add_container("i", n_type);

    // Add block
    auto& block = builder.add_block(builder.subject().root());

    auto& access_in = builder.add_access(block, "arg1");
    auto& access_out = builder.add_access(block, "t1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")});

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, block));

    auto arguments = analysis.arguments(analysis_manager, block);
    EXPECT_EQ(arguments.size(), 1);
    EXPECT_TRUE(arguments.contains("arg1"));
    EXPECT_TRUE(arguments.at("arg1").is_input);

    auto locals = analysis.locals(analysis_manager, block);
    EXPECT_EQ(locals.size(), 2);
    EXPECT_TRUE(locals.contains("t1"));
    EXPECT_TRUE(locals.contains("i"));

    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, block, false));
    auto arg_sizes = analysis.argument_sizes(analysis_manager, block, false);
    EXPECT_EQ(arg_sizes.size(), 1);
    EXPECT_TRUE(arg_sizes.contains("arg1"));
    EXPECT_TRUE(symbolic::
                    eq(arg_sizes.at("arg1"),
                       symbolic::mul(symbolic::integer(4), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))))
    );
}

TEST(ArgumentsAnalysisTest, Block_Arguments_Pointers) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Add containers
    types::Scalar int_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(int_type);
    builder.add_container("arg1", pointer_type, true);
    builder.add_container("t1", pointer_type);
    builder.add_container("i", int_type);

    // Add block
    auto& block = builder.add_block(builder.subject().root());

    auto& access_in = builder.add_access(block, "arg1");
    auto& access_out = builder.add_access(block, "t1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")});

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, block));

    auto arguments = analysis.arguments(analysis_manager, block);
    EXPECT_EQ(arguments.size(), 1);
    EXPECT_TRUE(arguments.contains("arg1"));
    EXPECT_TRUE(arguments.at("arg1").is_input);

    auto locals = analysis.locals(analysis_manager, block);
    EXPECT_EQ(locals.size(), 2);
    EXPECT_TRUE(locals.contains("t1"));
    EXPECT_TRUE(locals.contains("i"));

    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, block, false));
    auto arg_sizes = analysis.argument_sizes(analysis_manager, block, false);
    EXPECT_EQ(arg_sizes.size(), 1);
    EXPECT_TRUE(arg_sizes.contains("arg1"));
    EXPECT_TRUE(symbolic::
                    eq(arg_sizes.at("arg1"),
                       symbolic::mul(symbolic::integer(4), symbolic::add(symbolic::symbol("i"), symbolic::integer(1))))
    );
}

TEST(ArgumentsAnalysisTest, Sequence_Blocks) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Add containers
    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("arg1", int_type, true);
    builder.add_container("t1", int_type);
    builder.add_container("t2", int_type);

    // Add blocks
    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& block2 = builder.add_block(root);

    // Block 1
    {
        auto& access_in = builder.add_access(block1, "arg1");
        auto& access_out = builder.add_access(block1, "t1");
        auto& tasklet = builder.add_tasklet(block1, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block1, access_in, tasklet, "_in", {});
        builder.add_computational_memlet(block1, tasklet, "_out", access_out, {});
    }

    // Block 2
    {
        auto& access_in = builder.add_access(block2, "t1");
        auto& access_out = builder.add_access(block2, "t2");
        auto& tasklet = builder.add_tasklet(block2, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block2, access_in, tasklet, "_in", {});
        builder.add_computational_memlet(block2, tasklet, "_out", access_out, {});
    }

    auto& sdfg = builder.subject();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check Block 1
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, block1));

    auto arguments1 = analysis.arguments(analysis_manager, block1);
    EXPECT_EQ(arguments1.size(), 2);
    EXPECT_TRUE(arguments1.contains("arg1"));
    EXPECT_TRUE(arguments1.at("arg1").is_input);
    EXPECT_TRUE(arguments1.contains("t1"));
    EXPECT_TRUE(arguments1.at("t1").is_output);

    auto locals1 = analysis.locals(analysis_manager, block1);
    EXPECT_EQ(locals1.size(), 0);

    // Check Block 2
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, block2));
    auto arguments2 = analysis.arguments(analysis_manager, block2);
    EXPECT_EQ(arguments2.size(), 1);
    EXPECT_TRUE(arguments2.contains("t1"));
    EXPECT_TRUE(arguments2.at("t1").is_input);

    auto locals2 = analysis.locals(analysis_manager, block2);
    EXPECT_EQ(locals2.size(), 1);
    EXPECT_TRUE(locals2.contains("t2"));

    // Check overall
    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, root, false));
    auto arg_sizes = analysis.argument_sizes(analysis_manager, root, false);
    EXPECT_EQ(arg_sizes.size(), 1);
    EXPECT_TRUE(arg_sizes.contains("arg1"));
    EXPECT_TRUE(symbolic::eq(arg_sizes.at("arg1"), symbolic::integer(4)));
}

TEST(ArgumentsAnalysisTest, Loop_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Add containers
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Scalar n_type(types::PrimitiveType::Int32);
    types::Array array_type(float_type, {symbolic::symbol("N")});
    builder.add_container("arg1", array_type, true);
    builder.add_container("t1", array_type);
    builder.add_container("i", n_type);
    builder.add_container("N", n_type);

    // Add block
    auto& root = builder.subject().root();

    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N"));
    auto increment = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& loop = builder.add_for(root, symbolic::symbol("i"), condition, init, increment);
    auto& block = builder.add_block(loop.root());

    auto& access_in = builder.add_access(block, "arg1");
    auto& access_out = builder.add_access(block, "t1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")});

    auto& sdfg = builder.subject();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, loop));

    auto arguments = analysis.arguments(analysis_manager, loop);
    EXPECT_EQ(arguments.size(), 1);
    EXPECT_TRUE(arguments.contains("arg1"));
    EXPECT_TRUE(arguments.at("arg1").is_input);

    auto locals = analysis.locals(analysis_manager, loop);
    EXPECT_EQ(locals.size(), 3);
    EXPECT_TRUE(locals.contains("t1"));
    EXPECT_TRUE(locals.contains("i"));
    EXPECT_TRUE(locals.contains("N"));

    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, loop, false));
    auto arg_sizes = analysis.argument_sizes(analysis_manager, loop, false);
    EXPECT_EQ(arg_sizes.size(), 1);
    EXPECT_TRUE(arg_sizes.contains("arg1"));
    EXPECT_TRUE(symbolic::eq(arg_sizes.at("arg1"), symbolic::mul(symbolic::symbol("N"), symbolic::integer(4))));
}

TEST(ArgumentsAnalysisTest, Map_Array) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Add containers
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Scalar n_type(types::PrimitiveType::Int32);
    types::Array array_type(float_type, {symbolic::symbol("N")});
    builder.add_container("arg1", array_type, true);
    builder.add_container("t1", array_type);
    builder.add_container("i", n_type);
    builder.add_container("N", n_type);

    // Add block
    auto& root = builder.subject().root();

    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N"));
    auto increment = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& loop =
        builder.add_map(root, symbolic::symbol("i"), condition, init, increment, ScheduleType_Sequential::create());
    auto& block = builder.add_block(loop.root());

    auto& access_in = builder.add_access(block, "arg1");
    auto& access_out = builder.add_access(block, "t1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")});

    auto& sdfg = builder.subject();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, loop));

    auto arguments = analysis.arguments(analysis_manager, loop);
    EXPECT_EQ(arguments.size(), 1);
    EXPECT_TRUE(arguments.contains("arg1"));
    EXPECT_TRUE(arguments.at("arg1").is_input);

    auto locals = analysis.locals(analysis_manager, loop);
    EXPECT_EQ(locals.size(), 3);
    EXPECT_TRUE(locals.contains("t1"));
    EXPECT_TRUE(locals.contains("i"));
    EXPECT_TRUE(locals.contains("N"));

    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, loop, false));
    auto arg_sizes = analysis.argument_sizes(analysis_manager, loop, false);
    EXPECT_EQ(arg_sizes.size(), 1);
    EXPECT_TRUE(arg_sizes.contains("arg1"));
    EXPECT_TRUE(symbolic::eq(arg_sizes.at("arg1"), symbolic::mul(symbolic::symbol("N"), symbolic::integer(4))));
}

TEST(ArgumentsAnalysisTest, Loop_Pointer) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Add containers
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Scalar n_type(types::PrimitiveType::Int32);
    types::Pointer array_type(float_type);
    builder.add_container("arg1", array_type, true);
    builder.add_container("t1", array_type);
    builder.add_container("i", n_type);
    builder.add_container("N", n_type);

    // Add block
    auto& root = builder.subject().root();

    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N"));
    auto increment = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& loop = builder.add_for(root, symbolic::symbol("i"), condition, init, increment);
    auto& block = builder.add_block(loop.root());

    auto& access_in = builder.add_access(block, "arg1");
    auto& access_out = builder.add_access(block, "t1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")});

    auto& sdfg = builder.subject();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, loop));

    auto arguments = analysis.arguments(analysis_manager, loop);
    EXPECT_EQ(arguments.size(), 1);
    EXPECT_TRUE(arguments.contains("arg1"));
    EXPECT_TRUE(arguments.at("arg1").is_input);

    auto locals = analysis.locals(analysis_manager, loop);
    EXPECT_EQ(locals.size(), 3);
    EXPECT_TRUE(locals.contains("t1"));
    EXPECT_TRUE(locals.contains("i"));
    EXPECT_TRUE(locals.contains("N"));

    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, loop, false));
    auto arg_sizes = analysis.argument_sizes(analysis_manager, loop, false);
    EXPECT_EQ(arg_sizes.size(), 1);
    EXPECT_TRUE(arg_sizes.contains("arg1"));
    EXPECT_TRUE(symbolic::eq(arg_sizes.at("arg1"), symbolic::mul(symbolic::symbol("N"), symbolic::integer(4))));
}

TEST(ArgumentsAnalysisTest, Map_Pointer) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Add containers
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Scalar n_type(types::PrimitiveType::Int32);
    types::Pointer array_type(float_type);
    builder.add_container("arg1", array_type, true);
    builder.add_container("t1", array_type);
    builder.add_container("i", n_type);
    builder.add_container("N", n_type);

    // Add block
    auto& root = builder.subject().root();

    auto init = symbolic::integer(0);
    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::symbol("N"));
    auto increment = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& loop =
        builder.add_map(root, symbolic::symbol("i"), condition, init, increment, ScheduleType_Sequential::create());
    auto& block = builder.add_block(loop.root());

    auto& access_in = builder.add_access(block, "arg1");
    auto& access_out = builder.add_access(block, "t1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")});

    auto& sdfg = builder.subject();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, loop));

    auto arguments = analysis.arguments(analysis_manager, loop);
    EXPECT_EQ(arguments.size(), 1);
    EXPECT_TRUE(arguments.contains("arg1"));
    EXPECT_TRUE(arguments.at("arg1").is_input);

    auto locals = analysis.locals(analysis_manager, loop);
    EXPECT_EQ(locals.size(), 3);
    EXPECT_TRUE(locals.contains("t1"));
    EXPECT_TRUE(locals.contains("i"));
    EXPECT_TRUE(locals.contains("N"));

    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, loop, false));
    auto arg_sizes = analysis.argument_sizes(analysis_manager, loop, false);
    EXPECT_EQ(arg_sizes.size(), 1);
    EXPECT_TRUE(arg_sizes.contains("arg1"));
    EXPECT_TRUE(symbolic::eq(arg_sizes.at("arg1"), symbolic::mul(symbolic::symbol("N"), symbolic::integer(4))));
}

TEST(ArgumentsAnalysisTest, Block_Arguments_ReferenceScalar) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Build SDFG with normal scalar types first
    types::Scalar double_type(types::PrimitiveType::Double);
    builder.add_container("arg1", double_type, true);
    builder.add_container("t1", double_type);

    // Add block with dataflow
    auto& block = builder.add_block(builder.subject().root());

    auto& access_in = builder.add_access(block, "arg1");
    auto& access_out = builder.add_access(block, "t1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {});

    // Change container types to Reference(Scalar) (as produced by CutoutSerializer)
    codegen::Reference ref_type(double_type);
    builder.change_type("arg1", ref_type);
    builder.change_type("t1", ref_type);

    auto& sdfg = builder.subject();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check: Reference(Scalar) should be treated as scalar
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, block));

    auto arguments = analysis.arguments(analysis_manager, block);
    EXPECT_EQ(arguments.size(), 1);
    EXPECT_TRUE(arguments.contains("arg1"));
    EXPECT_TRUE(arguments.at("arg1").is_scalar);
    EXPECT_FALSE(arguments.at("arg1").is_ptr);

    auto locals = analysis.locals(analysis_manager, block);
    EXPECT_EQ(locals.size(), 1);
    EXPECT_TRUE(locals.contains("t1"));

    EXPECT_TRUE(analysis.argument_size_known(analysis_manager, block, false));
    auto arg_sizes = analysis.argument_sizes(analysis_manager, block, false);
    EXPECT_EQ(arg_sizes.size(), 1);
    EXPECT_TRUE(arg_sizes.contains("arg1"));
    EXPECT_TRUE(symbolic::eq(arg_sizes.at("arg1"), symbolic::integer(8)));
}

TEST(ArgumentsAnalysisTest, Block_Arguments_ReferencePointer) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    // Build SDFG with normal types first
    types::Scalar float_type(types::PrimitiveType::Float);
    types::Scalar n_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(float_type);
    builder.add_container("arg1", pointer_type, true);
    builder.add_container("t1", pointer_type);
    builder.add_container("i", n_type);

    // Add block with dataflow
    auto& block = builder.add_block(builder.subject().root());

    auto& access_in = builder.add_access(block, "arg1");
    auto& access_out = builder.add_access(block, "t1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_in, tasklet, "_in", {symbolic::symbol("i")});
    builder.add_computational_memlet(block, tasklet, "_out", access_out, {symbolic::symbol("i")});

    // Change container types to Reference(Pointer) (as produced by CutoutSerializer)
    codegen::Reference ref_ptr_type(pointer_type);
    builder.change_type("arg1", ref_ptr_type);
    builder.change_type("t1", ref_ptr_type);

    auto& sdfg = builder.subject();

    // Analysis
    analysis::AnalysisManager analysis_manager(sdfg);
    auto& analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    // Check: Reference(Pointer) should be treated as pointer
    EXPECT_TRUE(analysis.inferred_types(analysis_manager, block));

    auto arguments = analysis.arguments(analysis_manager, block);
    EXPECT_EQ(arguments.size(), 1);
    EXPECT_TRUE(arguments.contains("arg1"));
    EXPECT_FALSE(arguments.at("arg1").is_scalar);
    EXPECT_TRUE(arguments.at("arg1").is_ptr);

    auto locals = analysis.locals(analysis_manager, block);
    EXPECT_EQ(locals.size(), 2);
    EXPECT_TRUE(locals.contains("t1"));
    EXPECT_TRUE(locals.contains("i"));
}
