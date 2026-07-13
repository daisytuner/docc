#include "sdfg/visitor/structured_sdfg_walker.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"

#include <iostream>

using namespace sdfg::visitor;
using namespace sdfg;

TEST(StructuredSDFGWalkerTest, WalksFullSDFG) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& sequence = builder.add_sequence(root);
    auto& block = builder.add_block(sequence);
    auto& if_else = builder.add_if_else(sequence);
    auto& branch_a = builder.add_case(if_else, symbolic::__true__());
    auto& assgn = builder.add_assignments(branch_a, {});
    auto& branch_b = builder.add_case(if_else, symbolic::__true__());
    auto& loop = builder.add_while(sequence);
    auto& cont = builder.add_continue(loop.root());
    auto& br = builder.add_break(loop.root());
    auto& for_l = builder.add_for(
        sequence,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::integer(0)),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& ret = builder.add_return(sequence, "");

    auto it = StructuredSDFGWalker::root(builder.subject());
    auto end = StructuredSDFGWalker::end();

    std::vector<std::pair<ControlFlowNode*, StructuredSDFGWalker::Scope>> order;

    while (it != end) {
        auto v = *it;
        order.push_back(std::make_pair(&v.first, v.second));
        // std::cout << "Node: " << v.first.element_id() << ": " << v.first.type_id() << ": " << v.second << std::endl;

        ++it;
    }

    EXPECT_EQ(order.size(), 24);
    EXPECT_EQ(order.at(0), std::make_pair(&root, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(1), std::make_pair(&sequence, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(2), std::make_pair(&block, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(3), std::make_pair(&if_else, StructuredSDFGWalker::Scope::IF_ENTRY));
    EXPECT_EQ(order.at(4), std::make_pair(&branch_a, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(5), std::make_pair(&assgn, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(6), std::make_pair(&branch_a, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(7), std::make_pair(&if_else, StructuredSDFGWalker::Scope::IF_NEXT_BRANCH));
    EXPECT_EQ(order.at(8), std::make_pair(&branch_b, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(9), std::make_pair(&branch_b, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(10), std::make_pair(&if_else, StructuredSDFGWalker::Scope::IF_EXIT));
    EXPECT_EQ(order.at(11), std::make_pair(&loop, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(12), std::make_pair(&loop.root(), StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(13), std::make_pair(&cont, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(14), std::make_pair(&br, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(15), std::make_pair(&loop.root(), StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(16), std::make_pair(&loop, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(17), std::make_pair(&for_l, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(18), std::make_pair(&for_l.root(), StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(19), std::make_pair(&for_l.root(), StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(20), std::make_pair(&for_l, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(21), std::make_pair(&ret, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(22), std::make_pair(&sequence, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(23), std::make_pair(&root, StructuredSDFGWalker::Scope::EXIT));
}

TEST(StructuredSDFGWalkerTest, CanWalkUntilTarget) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& sequence = builder.add_sequence(root);
    auto& block = builder.add_block(sequence);
    auto& if_else = builder.add_if_else(sequence);
    auto& branch_a = builder.add_case(if_else, symbolic::__true__());
    auto& assgn = builder.add_assignments(branch_a, {});
    auto& branch_b = builder.add_case(if_else, symbolic::__true__());
    auto& loop = builder.add_while(sequence);
    auto& cont = builder.add_continue(loop.root());
    auto& br = builder.add_break(loop.root());
    auto& for_l = builder.add_for(
        sequence,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::integer(0)),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& ret = builder.add_return(sequence, "");

    auto it = StructuredSDFGWalker::root(builder.subject());
    auto end = StructuredSDFGWalker::from_after(assgn);

    std::vector<std::pair<ControlFlowNode*, StructuredSDFGWalker::Scope>> order;

    while (it != end) {
        auto v = *it;
        order.push_back(std::make_pair(&v.first, v.second));
        std::cout << "Node: " << v.first.element_id() << ": " << v.first.type_id() << ": " << v.second << std::endl;

        ++it;
    }

    EXPECT_EQ(order.size(), 6);
    EXPECT_EQ(order.at(0), std::make_pair(&root, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(1), std::make_pair(&sequence, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(2), std::make_pair(&block, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(3), std::make_pair(&if_else, StructuredSDFGWalker::Scope::IF_ENTRY));
    EXPECT_EQ(order.at(4), std::make_pair(&branch_a, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(5), std::make_pair(&assgn, StructuredSDFGWalker::Scope::NONE));
}

TEST(StructuredSDFGWalkerTest, WalksFullSDFG_FromNested) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& sequence = builder.add_sequence(root);
    auto& block = builder.add_block(sequence);
    auto& if_else = builder.add_if_else(sequence);
    auto& branch_a = builder.add_case(if_else, symbolic::__true__());
    auto& assgn = builder.add_assignments(branch_a, {});
    auto& branch_b = builder.add_case(if_else, symbolic::__true__());
    auto& loop = builder.add_while(sequence);
    auto& cont = builder.add_continue(loop.root());
    auto& br = builder.add_break(loop.root());
    auto& for_l = builder.add_for(
        sequence,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::integer(0)),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& ret = builder.add_return(sequence, "");

    auto it = StructuredSDFGWalker::from_node(assgn);
    auto end = StructuredSDFGWalker::end();

    std::vector<std::pair<ControlFlowNode*, StructuredSDFGWalker::Scope>> order;

    while (it != end) {
        auto v = *it;
        order.push_back(std::make_pair(&v.first, v.second));
        // std::cout << "Node: " << v.first.element_id() << ": " << v.first.type_id() << ": " << v.second << std::endl;

        ++it;
    }

    EXPECT_EQ(order.size(), 19);
    EXPECT_EQ(order.at(0), std::make_pair(&assgn, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(1), std::make_pair(&branch_a, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(2), std::make_pair(&if_else, StructuredSDFGWalker::Scope::IF_NEXT_BRANCH));
    EXPECT_EQ(order.at(3), std::make_pair(&branch_b, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(4), std::make_pair(&branch_b, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(5), std::make_pair(&if_else, StructuredSDFGWalker::Scope::IF_EXIT));
    EXPECT_EQ(order.at(6), std::make_pair(&loop, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(7), std::make_pair(&loop.root(), StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(8), std::make_pair(&cont, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(9), std::make_pair(&br, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(10), std::make_pair(&loop.root(), StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(11), std::make_pair(&loop, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(12), std::make_pair(&for_l, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(13), std::make_pair(&for_l.root(), StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(14), std::make_pair(&for_l.root(), StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(15), std::make_pair(&for_l, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(16), std::make_pair(&ret, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(17), std::make_pair(&sequence, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(18), std::make_pair(&root, StructuredSDFGWalker::Scope::EXIT));
}

TEST(StructuredSDFGWalkerTest, WalksOnlyOut) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& sequence = builder.add_sequence(root);
    auto& block = builder.add_block(sequence);
    auto& if_else = builder.add_if_else(sequence);
    auto& branch_a = builder.add_case(if_else, symbolic::__true__());
    auto& assgn = builder.add_assignments(branch_a, {});
    auto& branch_b = builder.add_case(if_else, symbolic::__true__());
    auto& loop = builder.add_while(sequence);
    auto& cont = builder.add_continue(loop.root());
    auto& br = builder.add_break(loop.root());
    auto& for_l = builder.add_for(
        sequence,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::integer(0)),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& ret = builder.add_return(sequence, "");

    auto it = StructuredSDFGWalker::from_node(assgn);
    auto end = StructuredSDFGWalker::end();

    std::vector<std::pair<ControlFlowNode*, StructuredSDFGWalker::Scope>> order;

    while (it != end) {
        auto v = *it;
        order.push_back(std::make_pair(&v.first, v.second));
        // std::cout << "Node: " << v.first.element_id() << ": " << v.first.type_id() << ": " << v.second << std::endl;

        it.next_no_descend();
    }

    EXPECT_EQ(order.size(), 8);
    EXPECT_EQ(order.at(0), std::make_pair(&assgn, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(1), std::make_pair(&branch_a, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(2), std::make_pair(&if_else, StructuredSDFGWalker::Scope::IF_NEXT_BRANCH));
    EXPECT_EQ(order.at(3), std::make_pair(&loop, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(4), std::make_pair(&for_l, StructuredSDFGWalker::Scope::ENTRY));
    EXPECT_EQ(order.at(5), std::make_pair(&ret, StructuredSDFGWalker::Scope::NONE));
    EXPECT_EQ(order.at(6), std::make_pair(&sequence, StructuredSDFGWalker::Scope::EXIT));
    EXPECT_EQ(order.at(7), std::make_pair(&root, StructuredSDFGWalker::Scope::EXIT));
}

TEST(StructuredSDFGWalkerTest, CanSkipChildrenConditionally) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& sequence = builder.add_sequence(root);
    auto& block = builder.add_block(sequence);
    auto& if_else = builder.add_if_else(sequence);
    auto& branch_a = builder.add_case(if_else, symbolic::__true__());
    auto& assgn = builder.add_assignments(branch_a, {});
    auto& branch_b = builder.add_case(if_else, symbolic::__true__());
    auto& loop = builder.add_while(sequence);
    auto& cont = builder.add_continue(loop.root());
    auto& br = builder.add_break(loop.root());
    auto& for_l = builder.add_for(
        sequence,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::integer(0)),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );
    auto& ret = builder.add_return(sequence, "");

    auto it = StructuredSDFGWalker::root(builder.subject());
    auto end = StructuredSDFGWalker::end();

    auto resolve = [&](StructuredSDFGWalker::Iterator& it) {
        auto resolved = *it;
        return std::make_pair(&resolved.first, resolved.second);
    };

    EXPECT_EQ(resolve(it), std::make_pair(&root, StructuredSDFGWalker::Scope::ENTRY));
    it.next();
    EXPECT_EQ(resolve(it), std::make_pair(&sequence, StructuredSDFGWalker::Scope::ENTRY));
    it.next();
    EXPECT_EQ(resolve(it), std::make_pair(&block, StructuredSDFGWalker::Scope::NONE));
    it.next();
    EXPECT_EQ(resolve(it), std::make_pair(&if_else, StructuredSDFGWalker::Scope::IF_ENTRY));
    it.next_no_descend();
    EXPECT_EQ(resolve(it), std::make_pair(&loop, StructuredSDFGWalker::Scope::ENTRY));
    it.next_no_descend();
    EXPECT_EQ(resolve(it), std::make_pair(&for_l, StructuredSDFGWalker::Scope::ENTRY));
    it.next_no_descend();
    EXPECT_EQ(resolve(it), std::make_pair(&ret, StructuredSDFGWalker::Scope::NONE));
    it.next_no_descend();
    EXPECT_EQ(resolve(it), std::make_pair(&sequence, StructuredSDFGWalker::Scope::EXIT));
    it.next_no_descend();
    EXPECT_EQ(resolve(it), std::make_pair(&root, StructuredSDFGWalker::Scope::EXIT));
    it.next_no_descend();
    EXPECT_FALSE(it != end);
}
