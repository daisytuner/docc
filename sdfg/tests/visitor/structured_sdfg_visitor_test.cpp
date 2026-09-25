#include "sdfg/visitor/structured_sdfg_visitor.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"

using namespace sdfg;

class NoneVisitor : public visitor::StructuredSDFGVisitor {
public:
    NoneVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }
};

TEST(StructuredSDFGVisitorTest, None) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& sequence = builder.add_sequence(root);
    auto& block = builder.add_block(sequence);
    auto& if_else = builder.add_if_else(sequence);
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

    NoneVisitor visitor(builder, analysis_manager);
    EXPECT_FALSE(visitor.visit());
}

class BlockVisitor : public visitor::StructuredSDFGVisitor {
public:
    BlockVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }

    bool accept(structured_control_flow::Block& node) override {
        return true;
    };

    bool accept(structured_control_flow::Sequence& node) override {
        return false;
    };

    bool accept(structured_control_flow::IfElse& node) override {
        return false;
    };

    bool accept(structured_control_flow::While& node) override {
        return false;
    };

    bool accept(structured_control_flow::Return& node) override {
        return false;
    };

    bool accept(structured_control_flow::Continue& node) override {
        return false;
    };

    bool accept(structured_control_flow::Break& node) override {
        return false;
    };

    bool accept(structured_control_flow::For& node) override {
        return false;
    };

    bool accept(structured_control_flow::Map& node) override {
        return false;
    };
};

TEST(StructuredSDFGVisitorTest, Block) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);

    BlockVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}

class SequenceVisitor : public visitor::StructuredSDFGVisitor {
public:
    SequenceVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }

    bool accept(structured_control_flow::Block& node) override {
        return false;
    };

    bool accept(structured_control_flow::Sequence& node) override {
        return &node != &builder_.subject().root();
    };

    bool accept(structured_control_flow::IfElse& node) override {
        return false;
    };

    bool accept(structured_control_flow::While& node) override {
        return false;
    };

    bool accept(structured_control_flow::Return& node) override {
        return false;
    };

    bool accept(structured_control_flow::Continue& node) override {
        return false;
    };

    bool accept(structured_control_flow::Break& node) override {
        return false;
    };

    bool accept(structured_control_flow::For& node) override {
        return false;
    };

    bool accept(structured_control_flow::Map& node) override {
        return false;
    };
};

TEST(StructuredSDFGVisitorTest, Sequence) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& sequence = builder.add_sequence(root);

    SequenceVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}

class IfElseVisitor : public visitor::StructuredSDFGVisitor {
public:
    IfElseVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }

    bool accept(structured_control_flow::Block& node) override {
        return false;
    };

    bool accept(structured_control_flow::Sequence& node) override {
        return false;
    };

    bool accept(structured_control_flow::IfElse& node) override {
        return true;
    };

    bool accept(structured_control_flow::While& node) override {
        return false;
    };

    bool accept(structured_control_flow::Return& node) override {
        return false;
    };

    bool accept(structured_control_flow::Continue& node) override {
        return false;
    };

    bool accept(structured_control_flow::Break& node) override {
        return false;
    };

    bool accept(structured_control_flow::For& node) override {
        return false;
    };

    bool accept(structured_control_flow::Map& node) override {
        return false;
    };
};

TEST(StructuredSDFGVisitorTest, IfElse) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& ifelse = builder.add_if_else(root);

    IfElseVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}

class WhileVisitor : public visitor::StructuredSDFGVisitor {
public:
    WhileVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }

    bool accept(structured_control_flow::Block& node) override {
        return false;
    };

    bool accept(structured_control_flow::Sequence& node) override {
        return false;
    };

    bool accept(structured_control_flow::IfElse& node) override {
        return false;
    };

    bool accept(structured_control_flow::While& node) override {
        return true;
    };

    bool accept(structured_control_flow::Return& node) override {
        return false;
    };

    bool accept(structured_control_flow::Continue& node) override {
        return false;
    };

    bool accept(structured_control_flow::Break& node) override {
        return false;
    };

    bool accept(structured_control_flow::For& node) override {
        return false;
    };

    bool accept(structured_control_flow::Map& node) override {
        return false;
    };
};

TEST(StructuredSDFGVisitorTest, While) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& while_ = builder.add_while(root);

    WhileVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}

class ReturnVisitor : public visitor::StructuredSDFGVisitor {
public:
    ReturnVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }

    bool accept(structured_control_flow::Block& node) override {
        return false;
    };

    bool accept(structured_control_flow::Sequence& node) override {
        return false;
    };

    bool accept(structured_control_flow::IfElse& node) override {
        return false;
    };

    bool accept(structured_control_flow::While& node) override {
        return false;
    };

    bool accept(structured_control_flow::Return& node) override {
        return true;
    };

    bool accept(structured_control_flow::Continue& node) override {
        return false;
    };

    bool accept(structured_control_flow::Break& node) override {
        return false;
    };

    bool accept(structured_control_flow::For& node) override {
        return false;
    };

    bool accept(structured_control_flow::Map& node) override {
        return false;
    };
};

TEST(StructuredSDFGVisitorTest, Return) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& return_ = builder.add_return(root, "");

    ReturnVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}

class ContinueVisitor : public visitor::StructuredSDFGVisitor {
public:
    ContinueVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }

    bool accept(structured_control_flow::Block& node) override {
        return false;
    };

    bool accept(structured_control_flow::Sequence& node) override {
        return false;
    };

    bool accept(structured_control_flow::IfElse& node) override {
        return false;
    };

    bool accept(structured_control_flow::While& node) override {
        return false;
    };

    bool accept(structured_control_flow::Return& node) override {
        return false;
    };

    bool accept(structured_control_flow::Continue& node) override {
        return true;
    };

    bool accept(structured_control_flow::Break& node) override {
        return false;
    };

    bool accept(structured_control_flow::For& node) override {
        return false;
    };

    bool accept(structured_control_flow::Map& node) override {
        return false;
    };
};

TEST(StructuredSDFGVisitorTest, Continue) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& loop = builder.add_while(root);
    auto& continue_ = builder.add_continue(loop.root());

    ContinueVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}

class BreakVisitor : public visitor::StructuredSDFGVisitor {
public:
    BreakVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }

    bool accept(structured_control_flow::Block& node) override {
        return false;
    };

    bool accept(structured_control_flow::Sequence& node) override {
        return false;
    };

    bool accept(structured_control_flow::IfElse& node) override {
        return false;
    };

    bool accept(structured_control_flow::While& node) override {
        return false;
    };

    bool accept(structured_control_flow::Return& node) override {
        return false;
    };

    bool accept(structured_control_flow::Continue& node) override {
        return false;
    };

    bool accept(structured_control_flow::Break& node) override {
        return true;
    };

    bool accept(structured_control_flow::For& node) override {
        return false;
    };

    bool accept(structured_control_flow::Map& node) override {
        return false;
    };
};

TEST(StructuredSDFGVisitorTest, Break) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& loop = builder.add_while(root);
    auto& break_ = builder.add_break(loop.root());

    BreakVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}

class ForVisitor : public visitor::StructuredSDFGVisitor {
public:
    ForVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }

    bool accept(structured_control_flow::Block& node) override {
        return false;
    };

    bool accept(structured_control_flow::Sequence& node) override {
        return false;
    };

    bool accept(structured_control_flow::IfElse& node) override {
        return false;
    };

    bool accept(structured_control_flow::While& node) override {
        return false;
    };

    bool accept(structured_control_flow::Return& node) override {
        return false;
    };

    bool accept(structured_control_flow::Continue& node) override {
        return false;
    };

    bool accept(structured_control_flow::Break& node) override {
        return false;
    };

    bool accept(structured_control_flow::For& node) override {
        return true;
    };

    bool accept(structured_control_flow::Map& node) override {
        return false;
    };
};

TEST(StructuredSDFGVisitorTest, For) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& for_ = builder.add_for(
        root,
        symbolic::symbol("i"),
        symbolic::Le(symbolic::symbol("i"), symbolic::integer(0)),
        symbolic::integer(1),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1))
    );

    ForVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}

class MapVisitor : public visitor::StructuredSDFGVisitor {
public:
    MapVisitor(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        : visitor::StructuredSDFGVisitor(builder, analysis_manager) {
    }

    bool accept(structured_control_flow::Block& node) override {
        return false;
    };

    bool accept(structured_control_flow::Sequence& node) override {
        return false;
    };

    bool accept(structured_control_flow::IfElse& node) override {
        return false;
    };

    bool accept(structured_control_flow::While& node) override {
        return false;
    };

    bool accept(structured_control_flow::Return& node) override {
        return false;
    };

    bool accept(structured_control_flow::Continue& node) override {
        return false;
    };

    bool accept(structured_control_flow::Break& node) override {
        return false;
    };

    bool accept(structured_control_flow::For& node) override {
        return false;
    };

    bool accept(structured_control_flow::Map& node) override {
        return true;
    };
};

TEST(StructuredSDFGVisitorTest, Map) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    analysis::AnalysisManager analysis_manager(builder.subject());

    auto& root = builder.subject().root();

    auto& kernel = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    MapVisitor visitor(builder, analysis_manager);
    EXPECT_TRUE(visitor.visit());
}
