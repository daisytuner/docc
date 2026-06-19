#include "sdfg/builder/structured_sdfg_builder.h"

#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/library_nodes/call_node.h"
#include "sdfg/element.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(StructuredSDFGBuilderTest, Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 0);
    EXPECT_EQ(sdfg->root().get_parent(), nullptr);
}

TEST(StructuredSDFGBuilderTest, AddBlock) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);
    EXPECT_EQ(root.get_parent(), nullptr);

    auto& block = builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);
    EXPECT_EQ(block.get_parent(), &root);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &block);
    EXPECT_EQ(child.second.size(), 1);
    EXPECT_EQ(child.first.get_parent(), &sdfg->root());
}

TEST(StructuredSDFGBuilderTest, AddBlockBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block_base.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);
    EXPECT_EQ(block_base.get_parent(), &root);

    auto& block = builder.add_block_before(root, block_base, {}, {});
    EXPECT_EQ(block.element_id(), 3);
    EXPECT_EQ(root.at(0).second.element_id(), 4);
    EXPECT_EQ(block.get_parent(), &root);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &block);
    EXPECT_EQ(child.second.size(), 0);
}

TEST(StructuredSDFGBuilderTest, AddBlockAfter) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block_base.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);
    EXPECT_EQ(block_base.get_parent(), &root);

    auto& block_base2 =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block_base2.element_id(), 3);
    EXPECT_EQ(root.at(1).second.element_id(), 4);
    EXPECT_EQ(block_base2.get_parent(), &root);

    auto& block = builder.add_block_after(root, block_base, {}, {});
    EXPECT_EQ(block.element_id(), 5);
    EXPECT_EQ(root.at(1).second.element_id(), 6);
    EXPECT_EQ(block.get_parent(), &root);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 3);

    auto child = sdfg->root().at(1);
    EXPECT_EQ(&child.first, &block);
    EXPECT_EQ(child.second.size(), 0);
}

TEST(StructuredSDFGBuilderTest, AddLibraryNode) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block = builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    EXPECT_EQ(block.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);
    EXPECT_EQ(block.get_parent(), &root);

    auto& lib_node = builder.add_library_node<data_flow::BarrierLocalNode>(block, DebugInfo());
    EXPECT_EQ(lib_node.element_id(), 3);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &block);
    EXPECT_EQ(child.second.size(), 1);

    EXPECT_EQ(block.dataflow().nodes().size(), 1);
    EXPECT_EQ(block.dataflow().edges().size(), 0);
    EXPECT_EQ(&(*block.dataflow().nodes().begin()), &lib_node);
    EXPECT_EQ(lib_node.code(), data_flow::LibraryNodeType_BarrierLocal);
    EXPECT_TRUE(dynamic_cast<data_flow::BarrierLocalNode*>(&lib_node));
}

TEST(StructuredSDFGBuilderTest, AddIfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& if_else = builder.add_if_else(root);
    EXPECT_EQ(if_else.element_id(), 1);
    EXPECT_EQ(root.at(0).second.element_id(), 2);
    EXPECT_EQ(if_else.get_parent(), &root);

    auto& true_case = builder.add_case(if_else, symbolic::__true__());
    EXPECT_EQ(true_case.element_id(), 3);
    EXPECT_EQ(true_case.get_parent(), &if_else);

    auto& false_case = builder.add_case(if_else, symbolic::__false__());
    EXPECT_EQ(false_case.element_id(), 4);
    EXPECT_EQ(false_case.get_parent(), &if_else);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &if_else);

    EXPECT_EQ(if_else.size(), 2);
    EXPECT_EQ(&if_else.at(0).first, &true_case);
    EXPECT_TRUE(symbolic::is_true(if_else.at(0).second));
    EXPECT_EQ(&if_else.at(1).first, &false_case);
    EXPECT_TRUE(symbolic::is_false(if_else.at(1).second));
}

TEST(StructuredSDFGBuilderTest, AddIfElseBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    auto& if_else = builder.add_if_else_before(root, block_base, {}, {});
    auto& true_case = builder.add_case(if_else, symbolic::__true__());
    auto& false_case = builder.add_case(if_else, symbolic::__false__());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &if_else);

    EXPECT_EQ(if_else.size(), 2);
    EXPECT_EQ(&if_else.at(0).first, &true_case);
    EXPECT_TRUE(symbolic::is_true(if_else.at(0).second));
    EXPECT_EQ(&if_else.at(1).first, &false_case);
    EXPECT_TRUE(symbolic::is_false(if_else.at(1).second));
}

TEST(StructuredSDFGBuilderTest, addWhile) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_while(root);
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);
    EXPECT_EQ(scope.get_parent(), &root);
    EXPECT_EQ(scope.root().get_parent(), &scope);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);
    EXPECT_EQ(body.get_parent(), &scope.root());

    auto& break_state = builder.add_break(scope.root());
    EXPECT_EQ(break_state.element_id(), 6);

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 2);
}

TEST(StructuredSDFGBuilderTest, addFor) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);
    EXPECT_EQ(scope.get_parent(), &root);
    EXPECT_EQ(scope.root().get_parent(), &scope);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);
    EXPECT_EQ(body.get_parent(), &scope.root());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addFor_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        {{symbolic::symbol("i"), symbolic::zero()}}
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);
    EXPECT_EQ(scope.get_parent(), &root);
    EXPECT_EQ(scope.root().get_parent(), &scope);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);
    EXPECT_EQ(body.get_parent(), &scope.root());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);
    EXPECT_EQ(sdfg->root().at(0).second.assignments().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addMap) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);
    EXPECT_EQ(scope.get_parent(), &root);
    EXPECT_EQ(scope.root().get_parent(), &scope);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);
    EXPECT_EQ(body.get_parent(), &scope.root());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto map = dynamic_cast<structured_control_flow::Map*>(&sdfg->root().at(0).first);
    EXPECT_TRUE(map);
    EXPECT_EQ(sdfg->root().at(0).second.assignments().size(), 0);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addMap_Transition) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& scope = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create(),
        {{symbolic::symbol("i"), symbolic::zero()}}
    );
    EXPECT_EQ(scope.element_id(), 1);
    EXPECT_EQ(scope.root().element_id(), 2);
    EXPECT_EQ(root.at(0).second.element_id(), 3);
    EXPECT_EQ(scope.get_parent(), &root);
    EXPECT_EQ(scope.root().get_parent(), &scope);

    auto& body = builder.add_block(scope.root());
    EXPECT_EQ(body.element_id(), 4);
    EXPECT_EQ(body.get_parent(), &scope.root());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 1);

    auto map = dynamic_cast<structured_control_flow::Map*>(&sdfg->root().at(0).first);
    EXPECT_TRUE(map);
    EXPECT_EQ(sdfg->root().at(0).second.assignments().size(), 1);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addForBefore) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    auto& scope = builder.add_for_before(
        root,
        block_base,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = builder.add_block(scope.root());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 2);

    auto child = sdfg->root().at(0);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}

TEST(StructuredSDFGBuilderTest, addForAfter) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int64));
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);

    auto& root = builder.subject().root();
    auto& block_base =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    auto& block_base2 =
        builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});
    auto& scope = builder.add_for_after(
        root,
        block_base,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& body = builder.add_block(scope.root());

    auto sdfg = builder.move();

    EXPECT_EQ(sdfg->name(), "sdfg_1");
    EXPECT_EQ(sdfg->root().size(), 3);

    auto child = sdfg->root().at(1);
    EXPECT_EQ(&child.first, &scope);
    EXPECT_EQ(scope.root().size(), 1);
}


TEST(StructuredSDFGBuilderTest, FindElementById_Root) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();

    // Test

    EXPECT_EQ(builder.find_element_by_id(root.element_id()), &root);
}

TEST(StructuredSDFGBuilderTest, FindElementById_Block) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    // Test

    EXPECT_EQ(builder.find_element_by_id(block.element_id()), &block);
}

TEST(StructuredSDFGBuilderTest, ClearNode_AccessNode_Unused) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block = builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});

    auto& in_node = builder.add_access(block, "N");
    auto& out_node = builder.add_access(block, "N");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, in_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {});

    dump_sdfg(builder.subject(), "0-before");

    builder.clear_node(block, out_node);

    dump_sdfg(builder.subject(), "1-after");

    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);
}

TEST(StructuredSDFGBuilderTest, ClearNode_AccessNode_Used) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block = builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});

    auto& in_node = builder.add_access(block, "N");
    auto& out_node = builder.add_access(block, "N");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, in_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {});

    auto& out_node2 = builder.add_access(block, "N");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, out_node, tasklet2, "_in", {});
    builder.add_computational_memlet(block, tasklet2, "_out", out_node2, {});

    dump_sdfg(builder.subject(), "0-before");

    builder.clear_node(block, out_node);

    dump_sdfg(builder.subject(), "1-after");

    EXPECT_EQ(block.dataflow().nodes().size(), 3);
    EXPECT_EQ(block.dataflow().edges().size(), 2);
}

TEST(StructuredSDFGBuilderTest, ClearNode_AccessNode_Unused_Diamond) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("N", desc);
    builder.add_container("tmp1", desc);
    builder.add_container("tmp2", desc);

    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block = builder.add_block(root, control_flow::Assignments{{symbolic::symbol("N"), SymEngine::integer(10)}});

    auto& in_node = builder.add_access(block, "N");
    auto& out_node = builder.add_access(block, "tmp1");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_add, "_out", {"a", "b"});
    builder.add_computational_memlet(block, in_node, tasklet, "a", {});
    builder.add_computational_memlet(block, in_node, tasklet, "b", {});
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {});

    dump_sdfg(builder.subject(), "0-before");

    builder.clear_node(block, out_node);

    dump_sdfg(builder.subject(), "1-after");
    EXPECT_EQ(block.dataflow().nodes().size(), 0);
    EXPECT_EQ(block.dataflow().edges().size(), 0);
}

TEST(StructuredSDFGBuilderTest, ClearNode_Dont_Remove_Write_Nodes) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    types::Pointer desc_ptr(desc);
    builder.add_container("ptr", desc_ptr, true);
    builder.add_container("tmp1", desc);
    builder.add_container("N", desc, true);


    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block = builder.add_block(root);

    auto& in_node = builder.add_access(block, "N");
    auto& out_ptr_node = builder.add_access(block, "ptr");
    builder.add_dereference_memlet(block, in_node, out_ptr_node, false, desc_ptr);

    auto& out_node = builder.add_access(block, "tmp1");
    builder.add_dereference_memlet(block, out_ptr_node, out_node, true, desc_ptr);

    dump_sdfg(builder.subject(), "0-before");

    builder.clear_node(block, out_node);

    dump_sdfg(builder.subject(), "1-after");
    EXPECT_EQ(block.dataflow().nodes().size(), 2);
    EXPECT_EQ(block.dataflow().edges().size(), 1);
}

TEST(StructuredSDFGBuilderTest, ClearNode_Leaves_Required_Out_Edges) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    types::Pointer desc_ptr(desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("result1", desc);
    builder.add_container("result2", desc);


    auto& root = builder.subject().root();
    EXPECT_EQ(root.element_id(), 0);

    auto& block = builder.add_block(root);

    auto& in_a_node = builder.add_access(block, "a");
    auto& in_b_node = builder.add_access(block, "b");
    auto& res1_node = builder.add_access(block, "result1");
    auto& res2_node = builder.add_access(block, "result2");
    auto function_type = types::Function(types::Scalar(types::Void));
    function_type.add_param(desc);
    function_type.add_param(desc);

    builder.add_external("blackbox_function", function_type, LinkageType_External);
    auto& black_fun = builder.add_library_node<data_flow::CallNode>(
        block,
        DebugInfo(),
        "blackbox_function",
        std::vector<std::string>{"res1", "res2"},
        std::vector<std::string>{"a", "b"}
    );

    builder.add_computational_memlet(block, in_a_node, black_fun, "a", {}, desc);
    builder.add_computational_memlet(block, in_b_node, black_fun, "b", {}, desc);
    builder.add_computational_memlet(block, black_fun, "res1", res1_node, {}, desc);
    builder.add_computational_memlet(block, black_fun, "res2", res2_node, {}, desc);

    builder.add_return(root, "result2");

    dump_sdfg(builder.subject(), "0-before");

    builder.subject().validate();

    builder.clear_node(block, res1_node);

    dump_sdfg(builder.subject(), "1-after");

    builder.subject().validate();

    EXPECT_EQ(block.dataflow().nodes().size(), 5);
    EXPECT_EQ(block.dataflow().edges().size(), 4);
    EXPECT_EQ(black_fun.outputs().size(), 2);

    auto& dflow = block.dataflow();
    EXPECT_EQ(dflow.out_degree(black_fun), 2);
}

TEST(StructuredSDFGBuilderTest, MoveChild) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& if_else = builder.add_if_else(root);
    auto& case_true = builder.add_case(if_else, symbolic::__true__());

    auto& source_block1 = builder.add_block(root);
    auto& source_block2 = builder.add_block(root);

    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(source_block1.get_parent(), &root);
    EXPECT_EQ(source_block2.get_parent(), &root);
    EXPECT_EQ(case_true.size(), 0);

    // move source_block1 to target_seq at the end
    builder.move_child(root, 1, case_true);

    EXPECT_EQ(root.size(), 2);
    EXPECT_EQ(if_else.size(), 1);
    EXPECT_EQ(if_else.at(0).first.size(), 1);
    EXPECT_EQ(if_else.at(0).first.at(0).first.element_id(), source_block1.element_id());
    EXPECT_EQ(source_block1.get_parent(), &case_true);

    // move source_block2 to target_seq at specific index
    builder.move_child(root, 1, case_true, 0);

    EXPECT_EQ(root.size(), 1);
    EXPECT_EQ(if_else.size(), 1);
    EXPECT_EQ(if_else.at(0).first.size(), 2);
    EXPECT_EQ(if_else.at(0).first.at(0).first.element_id(), source_block2.element_id());
    EXPECT_EQ(source_block2.get_parent(), &case_true);
    EXPECT_EQ(if_else.at(0).first.at(1).first.element_id(), source_block1.element_id());
    EXPECT_EQ(source_block1.get_parent(), &case_true);
}

TEST(StructuredSDFGBuilderTest, MoveChildren) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& root = builder.subject().root();
    auto& if_else1 = builder.add_if_else(root);
    auto& if_true1 = builder.add_case(if_else1, symbolic::__true__());
    auto& target_block = builder.add_block(if_true1);

    auto& if_else2 = builder.add_if_else(root);
    auto& if_true2 = builder.add_case(if_else2, symbolic::__true__());
    auto& source_block1 = builder.add_block(if_true2);
    auto& source_block2 = builder.add_block(if_true2);

    EXPECT_EQ(if_true2.size(), 2);
    EXPECT_EQ(if_true1.size(), 1);
    EXPECT_EQ(source_block1.get_parent(), &if_true2);
    EXPECT_EQ(source_block2.get_parent(), &if_true2);

    // move all children from if_true2 to if_true1 at the end
    builder.move_children(if_true2, if_true1);

    EXPECT_EQ(if_true2.size(), 0);
    EXPECT_EQ(if_true1.size(), 3);
    EXPECT_EQ(if_true1.at(0).first.element_id(), target_block.element_id());
    EXPECT_EQ(if_true1.at(1).first.element_id(), source_block1.element_id());
    EXPECT_EQ(if_true1.at(2).first.element_id(), source_block2.element_id());
    EXPECT_EQ(source_block1.get_parent(), &if_true1);
    EXPECT_EQ(source_block2.get_parent(), &if_true1);

    // move all children from target_seq to source_seq at index 0
    auto& source_block3 = builder.add_block(if_true2);
    builder.move_children(if_true1, if_true2, 0);

    EXPECT_EQ(if_true1.size(), 0);
    EXPECT_EQ(if_true2.size(), 4);
    EXPECT_EQ(if_true2.at(0).first.element_id(), target_block.element_id());
    EXPECT_EQ(if_true2.at(1).first.element_id(), source_block1.element_id());
    EXPECT_EQ(if_true2.at(2).first.element_id(), source_block2.element_id());
    EXPECT_EQ(if_true2.at(3).first.element_id(), source_block3.element_id());
    EXPECT_EQ(target_block.get_parent(), &if_true2);
    EXPECT_EQ(source_block1.get_parent(), &if_true2);
    EXPECT_EQ(source_block2.get_parent(), &if_true2);
}

TEST(StructuredSDFGBuilderTest, MergeSinks_SameContainer) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    // First write to A: B -> tasklet -> A
    auto& b_node = builder.add_access(block, "B");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, b_node, tasklet1, "_in", {});
    auto& a_node1 = builder.add_access(block, "A");
    builder.add_computational_memlet(block, tasklet1, "_out", a_node1, {});

    // Second write to A: C -> tasklet -> A
    auto& c_node = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, c_node, tasklet2, "_in", {});
    auto& a_node2 = builder.add_access(block, "A");
    builder.add_computational_memlet(block, tasklet2, "_out", a_node2, {});

    dump_sdfg(builder.subject(), "0-before");

    // Two separate sink access nodes both refer to "A"
    EXPECT_EQ(block.dataflow().sinks().size(), 2);
    EXPECT_EQ(block.dataflow().nodes().size(), 6);
    EXPECT_EQ(block.dataflow().edges().size(), 4);

    builder.merge_sinks(block);

    dump_sdfg(builder.subject(), "1-after");

    // The two "A" sinks are merged into a single node receiving both writes
    auto sinks = block.dataflow().sinks();
    EXPECT_EQ(sinks.size(), 1);
    auto* sink = dynamic_cast<data_flow::AccessNode*>(*sinks.begin());
    ASSERT_NE(sink, nullptr);
    EXPECT_EQ(sink->data(), "A");
    EXPECT_EQ(block.dataflow().in_degree(*sink), 2);
    EXPECT_EQ(block.dataflow().nodes().size(), 5);
    EXPECT_EQ(block.dataflow().edges().size(), 4);
}

TEST(StructuredSDFGBuilderTest, MergeSinks_DifferentContainers) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar desc(types::PrimitiveType::Int64);
    builder.add_container("A", desc);
    builder.add_container("B", desc);
    builder.add_container("C", desc);

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);

    // C -> tasklet -> A
    auto& c_node = builder.add_access(block, "C");
    auto& tasklet1 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, c_node, tasklet1, "_in", {});
    auto& a_node = builder.add_access(block, "A");
    builder.add_computational_memlet(block, tasklet1, "_out", a_node, {});

    // C -> tasklet -> B
    auto& c_node2 = builder.add_access(block, "C");
    auto& tasklet2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, c_node2, tasklet2, "_in", {});
    auto& b_node = builder.add_access(block, "B");
    builder.add_computational_memlet(block, tasklet2, "_out", b_node, {});

    dump_sdfg(builder.subject(), "0-before");

    EXPECT_EQ(block.dataflow().sinks().size(), 2);

    builder.merge_sinks(block);

    dump_sdfg(builder.subject(), "1-after");

    // Distinct containers must not be merged
    EXPECT_EQ(block.dataflow().sinks().size(), 2);
    EXPECT_EQ(block.dataflow().nodes().size(), 6);
    EXPECT_EQ(block.dataflow().edges().size(), 4);
}
