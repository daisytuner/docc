#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/symbolic/symbolic.h"
using namespace sdfg;

TEST(SymbolReplaceTest, AccessNodeTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    builder.add_container("scalar_1", types::Scalar(types::PrimitiveType::Int32));
    builder.add_container("scalar_2", types::Scalar(types::PrimitiveType::Int32));

    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    auto& access_node = builder.add_access(block, "scalar_1");

    EXPECT_EQ(access_node.data(), "scalar_1");

    access_node.replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_2"));

    EXPECT_EQ(access_node.data(), "scalar_2");
}

TEST(SymbolReplaceTest, BatchReplaceSequenceToSubsetTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar scalar(types::PrimitiveType::Int32);
    types::Pointer ptr(scalar);
    builder.add_container("ptr_1", ptr, true);
    builder.add_container("ptr_2", ptr, true);
    builder.add_container("a0", scalar, true);
    builder.add_container("a1", scalar, true);
    builder.add_container("b0", scalar, true);
    builder.add_container("b1", scalar, true);

    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    auto& src_node = builder.add_access(block, "ptr_1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    auto& dst_node = builder.add_access(block, "ptr_2");
    auto& memlet_src =
        builder.add_computational_memlet(block, src_node, tasklet, "_in", {symbolic::parse("3*a0+5-a1")}, ptr);
    auto& memlet_dst =
        builder.add_computational_memlet(block, tasklet, "_out", dst_node, {symbolic::parse("5+a0")}, ptr);

    symbolic::ExpressionMapping mapping{
        {symbolic::symbol("a0"), symbolic::symbol("b0")},
        {symbolic::symbol("a1"), symbolic::mul(symbolic::symbol("b1"), symbolic::integer(2))}
    };

    sdfg.root().replace(mapping);

    EXPECT_TRUE(symbolic::eq(memlet_src.subset().at(0), symbolic::parse("3*b0+5-(b1*2)")));
    EXPECT_TRUE(symbolic::eq(memlet_dst.subset().at(0), symbolic::parse("5+b0")));
}

/*
TEST(SymbolReplaceTest, TaskletTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("scalar_1", desc);
    builder.add_container("scalar_2", desc);
    builder.add_container("scalar_3", desc);
    builder.add_container("scalar_4", desc);

    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    auto& access_node_in = builder.add_access(block, "scalar_1");
    auto& access_node_out = builder.add_access(block, "scalar_2");

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", desc}, {{"_in", desc}});

    symbolic::Expression read_expr = symbolic::symbol("access_1");
    auto& memlet_in =
        builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {read_expr});

    symbolic::Expression write_expr = symbolic::symbol("access_2");
    symbolic::Expression write_expr2 = symbolic::symbol("access_3");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void",
                                          {write_expr, write_expr2});

    tasklet.replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_2"));

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));
}

TEST(SymbolReplaceTest, LibraryNodeTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    auto& access_node_in = builder.add_access(block, "scalar_1");
    auto& access_node_out = builder.add_access(block, "scalar_2");

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});

    symbolic::Expression read_expr = symbolic::symbol("access_1");
    auto& memlet_in =
        builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {read_expr});

    symbolic::Expression write_expr = symbolic::symbol("access_2");
    symbolic::Expression write_expr2 = symbolic::symbol("access_3");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void",
                                          {write_expr, write_expr2});

    auto& library_node =
        builder.add_library_node(block, DebugInfo());

    library_node.replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_42"));

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));
}

TEST(SymbolReplaceTest, MemletTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    auto& access_node_in = builder.add_access(block, "scalar_1");
    auto& access_node_out = builder.add_access(block, "scalar_2");

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});

    symbolic::Expression read_expr = symbolic::symbol("access_1");
    auto& memlet_in =
        builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {read_expr});

    symbolic::Expression write_expr = symbolic::symbol("access_2");
    symbolic::Expression write_expr2 = symbolic::symbol("access_3");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void",
                                          {write_expr, write_expr2});

    memlet_in.replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_42"));

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));

    memlet_out.replace(symbolic::symbol("access_3"), symbolic::symbol("access_42"));

    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), symbolic::symbol("access_42")));
}

TEST(SymbolReplaceTest, BlockNodeTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    auto& access_node = builder.add_access(block, "scalar_1");

    EXPECT_EQ(access_node.data(), "scalar_1");

    block.replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_42"));

    EXPECT_EQ(access_node.data(), "scalar_42");
}

TEST(SymbolReplaceTest, BlockWithMemletTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    auto& access_node_in = builder.add_access(block, "scalar_1");
    auto& access_node_out = builder.add_access(block, "scalar_2");

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});

    symbolic::Expression read_expr = symbolic::symbol("access_1");
    auto& memlet_in =
        builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {read_expr});

    symbolic::Expression write_expr = symbolic::symbol("access_2");
    symbolic::Expression write_expr2 = symbolic::symbol("access_3");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void",
                                          {write_expr, write_expr2});

    block.replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_42"));

    EXPECT_EQ(access_node_in.data(), "scalar_42");
    EXPECT_EQ(access_node_out.data(), "scalar_2");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));

    block.replace(symbolic::symbol("access_3"), symbolic::symbol("access_42"));

    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), symbolic::symbol("access_42")));
}

TEST(SymbolReplaceTest, SDFGTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    auto& access_node_in = builder.add_access(block, "scalar_1");
    auto& access_node_out = builder.add_access(block, "scalar_2");

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");

    auto& tasklet = builder.add_tasklet(block,
        data_flow::TaskletCode::assign,
        {"_out", types::Scalar(types::PrimitiveType::Int32)},
        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});

    symbolic::Expression read_expr = symbolic::symbol("access_1");
    auto& memlet_in = builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {
        read_expr
    });

    symbolic::Expression write_expr = symbolic::symbol("access_2");
    symbolic::Expression write_expr2 = symbolic::symbol("access_3");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void", {
        write_expr,
        write_expr2
    });

    sdfg.replace("scalar_1", "scalar_42");

    EXPECT_EQ(access_node_in.data(), "scalar_42");
    EXPECT_EQ(access_node_out.data(), "scalar_2");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));

    sdfg.replace("access_3", "access_42");

    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), symbolic::symbol("access_42")));
}

TEST(SymbolReplaceTest, ForLoopTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    symbolic::Symbol indvar = symbolic::symbol("indvar");
    symbolic::Symbol bound = symbolic::symbol("bound");
    symbolic::Condition condition = symbolic::Condition(symbolic::Lt(indvar, bound));
    symbolic::Expression init = symbolic::symbol("init");
    symbolic::Expression update = symbolic::add(symbolic::symbol("update"), indvar);
    auto& for_loop = builder.add_for(sdfg.root(), indvar, condition, init, update);

    auto& block = builder.add_block(for_loop.root());

    auto& access_node_in = builder.add_access(block, "scalar_1");
    auto& access_node_out = builder.add_access(block, "scalar_2");

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});

    symbolic::Expression read_expr = symbolic::symbol("access_1");
    auto& memlet_in =
        builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {read_expr});

    symbolic::Expression write_expr = symbolic::symbol("access_2");
    symbolic::Expression write_expr2 = symbolic::symbol("access_3");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void",
                                          {write_expr, write_expr2});

    for_loop.replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_42"));

    EXPECT_EQ(access_node_in.data(), "scalar_42");
    EXPECT_EQ(access_node_out.data(), "scalar_2");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));

    for_loop.replace(symbolic::symbol("access_3"), symbolic::symbol("access_42"));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), symbolic::symbol("access_42")));

    for_loop.replace(symbolic::symbol("indvar"), symbolic::symbol("indvar_42"));
    EXPECT_TRUE(symbolic::eq(for_loop.update(), symbolic::add(symbolic::symbol("update"),
                                                              symbolic::symbol("indvar_42"))));

    for_loop.replace(symbolic::symbol("bound"), symbolic::symbol("bound_42"));
    EXPECT_TRUE(symbolic::eq(for_loop.condition(), symbolic::Lt(symbolic::symbol("indvar_42"),
                                                                symbolic::symbol("bound_42"))));

    for_loop.replace(symbolic::symbol("init"), symbolic::symbol("init_42"));
    EXPECT_TRUE(symbolic::eq(for_loop.init(), symbolic::symbol("init_42")));
}

TEST(SymbolReplaceTest, SequenceTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    auto& access_node_in = builder.add_access(block, "scalar_1");
    auto& access_node_out = builder.add_access(block, "scalar_2");

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});

    symbolic::Expression read_expr = symbolic::symbol("access_1");
    auto& memlet_in =
        builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {read_expr});

    symbolic::Expression write_expr = symbolic::symbol("access_2");
    symbolic::Expression write_expr2 = symbolic::symbol("access_3");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void",
                                          {write_expr, write_expr2});

    sdfg.root().replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_42"));

    EXPECT_EQ(access_node_in.data(), "scalar_42");
    EXPECT_EQ(access_node_out.data(), "scalar_2");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));

    sdfg.root().replace(symbolic::symbol("access_3"), symbolic::symbol("access_42"));

    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), symbolic::symbol("access_42")));
}

TEST(SymbolReplaceTest, IfElseTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& if_else = builder.add_if_else(sdfg.root());

    symbolic::Condition condition1 =
        symbolic::Lt(symbolic::symbol("case_1"), symbolic::integer(42));
    symbolic::Condition condition2 =
        symbolic::Ge(symbolic::symbol("case_1"), symbolic::integer(42));
    auto& if_case = builder.add_case(if_else, condition1);

    auto& if_case2 = builder.add_case(if_else, condition2);

    auto& block = builder.add_block(if_case);

    auto& access_node_in = builder.add_access(block, "scalar_1");
    auto& access_node_out = builder.add_access(block, "scalar_2");

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});

    symbolic::Expression read_expr = symbolic::symbol("access_1");
    auto& memlet_in =
        builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {read_expr});

    symbolic::Expression write_expr = symbolic::symbol("access_2");
    symbolic::Expression write_expr2 = symbolic::symbol("access_3");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void",
                                          {write_expr, write_expr2});

    if_else.replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_42"));

    EXPECT_EQ(access_node_in.data(), "scalar_42");
    EXPECT_EQ(access_node_out.data(), "scalar_2");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));

    if_else.replace(symbolic::symbol("access_3"), symbolic::symbol("access_42"));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), symbolic::symbol("access_42")));

    if_else.replace(symbolic::symbol("case_1"), symbolic::symbol("case_42"));
    EXPECT_TRUE(symbolic::eq(if_else.at(0).second,
                             symbolic::Lt(symbolic::symbol("case_42"), symbolic::integer(42))));
}

TEST(SymbolReplaceTest, WhileTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& while_loop = builder.add_while(sdfg.root());

    auto& block = builder.add_block(while_loop.root());

    auto& access_node_in = builder.add_access(block, "scalar_1");
    auto& access_node_out = builder.add_access(block, "scalar_2");

    EXPECT_EQ(access_node_in.data(), "scalar_1");
    EXPECT_EQ(access_node_out.data(), "scalar_2");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});

    symbolic::Expression read_expr = symbolic::symbol("access_1");
    auto& memlet_in =
        builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {read_expr});

    symbolic::Expression write_expr = symbolic::symbol("access_2");
    symbolic::Expression write_expr2 = symbolic::symbol("access_3");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void",
                                          {write_expr, write_expr2});

    while_loop.replace(symbolic::symbol("scalar_1"), symbolic::symbol("scalar_42"));

    EXPECT_EQ(access_node_in.data(), "scalar_42");
    EXPECT_EQ(access_node_out.data(), "scalar_2");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));

    while_loop.replace(symbolic::symbol("access_3"), symbolic::symbol("access_42"));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), symbolic::symbol("access_42")));
}

TEST(SymbolReplaceTest, KernelConversionTest) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& while_loop = builder.add_while(sdfg.root());

    auto& block = builder.add_block(while_loop.root());

    auto& access_node_in = builder.add_access(block, "threadIdx.x");
    auto& access_node_out = builder.add_access(block, "threadIdx.y");

    EXPECT_EQ(access_node_in.data(), "threadIdx.x");
    EXPECT_EQ(access_node_out.data(), "threadIdx.y");

    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign,
                                        {"_out", types::Scalar(types::PrimitiveType::Int32)},
                                        {{"_in", types::Scalar(types::PrimitiveType::Int32)}});

    symbolic::Expression read_expr = symbolic::threadIdx_x();
    auto& memlet_in =
        builder.add_memlet(block, access_node_in, "void", tasklet, "_in", {read_expr});

    symbolic::Expression write_expr = symbolic::threadIdx_y();
    symbolic::Expression write_expr2 = symbolic::symbol("blockIdx.y");
    auto& memlet_out = builder.add_memlet(block, access_node_out, "_out", tasklet, "void",
                                          {write_expr, write_expr2});

    auto& kernel = builder.convert_into_kernel();

    kernel.replace(symbolic::symbol("__daisy_threadIdx_x_sdfg_1"),
                   symbolic::symbol("__daisy_threadIdx_x"));

    EXPECT_EQ(access_node_in.data(), "__daisy_threadIdx_x");
    EXPECT_EQ(access_node_out.data(), "__daisy_threadIdx_y_sdfg_1");
    read_expr = symbolic::symbol("__daisy_threadIdx_x");
    write_expr = symbolic::symbol("__daisy_threadIdx_y_sdfg_1");
    write_expr2 = symbolic::symbol("__daisy_blockIdx_y_sdfg_1");
    EXPECT_TRUE(symbolic::eq(memlet_in.subset().at(0), read_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(0), write_expr));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), write_expr2));

    kernel.replace(symbolic::symbol("__daisy_blockIdx_y_sdfg_1"),
                   symbolic::symbol("__daisy_blockIdx_y"));
    EXPECT_TRUE(symbolic::eq(memlet_out.subset().at(1), symbolic::symbol("__daisy_blockIdx_y")));
}
*/
