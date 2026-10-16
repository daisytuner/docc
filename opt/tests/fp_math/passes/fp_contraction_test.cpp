#include "sdfg/fp_math/passes/fp_contraction.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

// d = a * b + c * e: one of the two products is contracted, the other remains a multiplication.
TEST(FPContractionPassTest, TwoProducts) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    types::Scalar desc(types::PrimitiveType::Double);
    for (auto name : {"a", "b", "c", "e", "d"}) {
        builder.add_container(name, desc, true);
    }
    builder.add_container("t1", desc);
    builder.add_container("t2", desc);
    auto& block = builder.add_block(sdfg.root());

    auto product = [&](const std::string& x, const std::string& y, const std::string& t) -> data_flow::AccessNode& {
        auto& lhs = builder.add_access(block, x);
        auto& rhs = builder.add_access(block, y);
        auto& out = builder.add_access(block, t);
        auto& mul = builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block, lhs, mul, "_in1", {});
        builder.add_computational_memlet(block, rhs, mul, "_in2", {});
        builder.add_computational_memlet(block, mul, "_out", out, {});
        return out;
    };
    auto& t1 = product("a", "b", "t1");
    auto& t2 = product("c", "e", "t2");
    auto& d = builder.add_access(block, "d");
    auto& add = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, t1, add, "_in1", {});
    builder.add_computational_memlet(block, t2, add, "_in2", {});
    builder.add_computational_memlet(block, add, "_out", d, {});

    // A second block with an independent contraction.
    auto& block2 = builder.add_block(sdfg.root());
    builder.add_container("t3", desc);
    {
        auto& a = builder.add_access(block2, "a");
        auto& b = builder.add_access(block2, "b");
        auto& t3 = builder.add_access(block2, "t3");
        auto& c = builder.add_access(block2, "c");
        auto& e = builder.add_access(block2, "e");
        auto& mul = builder.add_tasklet(block2, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block2, a, mul, "_in1", {});
        builder.add_computational_memlet(block2, b, mul, "_in2", {});
        builder.add_computational_memlet(block2, mul, "_out", t3, {});
        auto& add2 = builder.add_tasklet(block2, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"});
        builder.add_computational_memlet(block2, c, add2, "_in1", {});
        builder.add_computational_memlet(block2, t3, add2, "_in2", {});
        builder.add_computational_memlet(block2, add2, "_out", e, {});
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::FPContractionPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));
    EXPECT_NO_THROW(sdfg.validate());

    auto count = [](structured_control_flow::Block& b, data_flow::TaskletCode code) {
        size_t n = 0;
        for (auto* tasklet : b.dataflow().tasklets()) {
            n += tasklet->code() == code;
        }
        return n;
    };
    EXPECT_EQ(block.dataflow().tasklets().size(), 2);
    EXPECT_EQ(count(block, data_flow::TaskletCode::fp_fma), 1);
    EXPECT_EQ(count(block, data_flow::TaskletCode::fp_mul), 1);
    EXPECT_FALSE(sdfg.exists("t1"));
    EXPECT_TRUE(sdfg.exists("t2"));

    EXPECT_EQ(block2.dataflow().tasklets().size(), 1);
    EXPECT_EQ(count(block2, data_flow::TaskletCode::fp_fma), 1);
    EXPECT_FALSE(sdfg.exists("t3"));

    EXPECT_FALSE(pass.run(builder, analysis_manager));
}
