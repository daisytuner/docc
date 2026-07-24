#include "sdfg/visitor/structured_sdfg_walker.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"

#include <iostream>

using namespace sdfg::visitor;
using namespace sdfg;

#define EXPECT_PAIR_EQ(val, p1, p2) \
    EXPECT_EQ((val).first, p1);     \
    EXPECT_EQ((val).second, p2)

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
        order.emplace_back(&v.first, v.second);
        // std::cout << "Node: " << v.first.element_id() << ": " << v.first.type_id() << ": " << v.second << std::endl;

        ++it;
    }

    EXPECT_EQ(order.size(), 24);
    EXPECT_PAIR_EQ(order.at(0), &root, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(1), &sequence, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(2), &block, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(3), &if_else, StructuredSDFGWalker::Scope::IF_ENTRY);
    EXPECT_PAIR_EQ(order.at(4), &branch_a, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(5), &assgn, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(6), &branch_a, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(7), &if_else, StructuredSDFGWalker::Scope::IF_NEXT_BRANCH);
    EXPECT_PAIR_EQ(order.at(8), &branch_b, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(9), &branch_b, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(10), &if_else, StructuredSDFGWalker::Scope::IF_EXIT);
    EXPECT_PAIR_EQ(order.at(11), &loop, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(12), &loop.root(), StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(13), &cont, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(14), &br, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(15), &loop.root(), StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(16), &loop, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(17), &for_l, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(18), &for_l.root(), StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(19), &for_l.root(), StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(20), &for_l, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(21), &ret, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(22), &sequence, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(23), &root, StructuredSDFGWalker::Scope::EXIT);
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
        order.emplace_back(&v.first, v.second);
        std::cout << "Node: " << v.first.element_id() << ": " << v.first.type_id() << ": " << v.second << std::endl;

        ++it;
    }

    EXPECT_EQ(order.size(), 6);
    EXPECT_PAIR_EQ(order.at(0), &root, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(1), &sequence, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(2), &block, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(3), &if_else, StructuredSDFGWalker::Scope::IF_ENTRY);
    EXPECT_PAIR_EQ(order.at(4), &branch_a, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(5), &assgn, StructuredSDFGWalker::Scope::NONE);
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
        order.emplace_back(&v.first, v.second);
        // std::cout << "Node: " << v.first.element_id() << ": " << v.first.type_id() << ": " << v.second << std::endl;

        ++it;
    }

    EXPECT_EQ(order.size(), 19);
    EXPECT_PAIR_EQ(order.at(0), &assgn, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(1), &branch_a, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(2), &if_else, StructuredSDFGWalker::Scope::IF_NEXT_BRANCH);
    EXPECT_PAIR_EQ(order.at(3), &branch_b, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(4), &branch_b, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(5), &if_else, StructuredSDFGWalker::Scope::IF_EXIT);
    EXPECT_PAIR_EQ(order.at(6), &loop, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(7), &loop.root(), StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(8), &cont, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(9), &br, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(10), &loop.root(), StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(11), &loop, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(12), &for_l, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(13), &for_l.root(), StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(14), &for_l.root(), StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(15), &for_l, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(16), &ret, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(17), &sequence, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(18), &root, StructuredSDFGWalker::Scope::EXIT);
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
        order.emplace_back(&v.first, v.second);
        // std::cout << "Node: " << v.first.element_id() << ": " << v.first.type_id() << ": " << v.second << std::endl;

        it.next_no_descend();
    }

    EXPECT_EQ(order.size(), 8);
    EXPECT_PAIR_EQ(order.at(0), &assgn, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(1), &branch_a, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(2), &if_else, StructuredSDFGWalker::Scope::IF_NEXT_BRANCH);
    EXPECT_PAIR_EQ(order.at(3), &loop, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(4), &for_l, StructuredSDFGWalker::Scope::ENTRY);
    EXPECT_PAIR_EQ(order.at(5), &ret, StructuredSDFGWalker::Scope::NONE);
    EXPECT_PAIR_EQ(order.at(6), &sequence, StructuredSDFGWalker::Scope::EXIT);
    EXPECT_PAIR_EQ(order.at(7), &root, StructuredSDFGWalker::Scope::EXIT);
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

    EXPECT_PAIR_EQ(resolve(it), &root, StructuredSDFGWalker::Scope::ENTRY);
    it.next();
    EXPECT_PAIR_EQ(resolve(it), &sequence, StructuredSDFGWalker::Scope::ENTRY);
    it.next();
    EXPECT_PAIR_EQ(resolve(it), &block, StructuredSDFGWalker::Scope::NONE);
    it.next();
    EXPECT_PAIR_EQ(resolve(it), &if_else, StructuredSDFGWalker::Scope::IF_ENTRY);
    it.next_no_descend();
    EXPECT_PAIR_EQ(resolve(it), &loop, StructuredSDFGWalker::Scope::ENTRY);
    it.next_no_descend();
    EXPECT_PAIR_EQ(resolve(it), &for_l, StructuredSDFGWalker::Scope::ENTRY);
    it.next_no_descend();
    EXPECT_PAIR_EQ(resolve(it), &ret, StructuredSDFGWalker::Scope::NONE);
    it.next_no_descend();
    EXPECT_PAIR_EQ(resolve(it), &sequence, StructuredSDFGWalker::Scope::EXIT);
    it.next_no_descend();
    EXPECT_PAIR_EQ(resolve(it), &root, StructuredSDFGWalker::Scope::EXIT);
    it.next_no_descend();
    EXPECT_FALSE(it != end);
}
