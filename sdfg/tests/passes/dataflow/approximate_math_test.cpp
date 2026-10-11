#include "sdfg/passes/dataflow/approximate_math.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

using namespace sdfg;

TEST(ApproximateMath, MarksAllCMathNodes) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("x", desc);
    builder.add_container("y", desc);
    builder.add_container("z", desc);

    auto& root = builder.subject().root();
    auto& block1 = builder.add_block(root);
    auto& exp_node = static_cast<math::cmath::CMathNode&>(builder.add_library_node<math::cmath::CMathNode>(
        block1, DebugInfo(), math::cmath::CMathFunction::exp, types::PrimitiveType::Float
    ));
    builder.add_computational_memlet(block1, builder.add_access(block1, "x"), exp_node, "_in1", {}, desc);
    builder.add_computational_memlet(block1, exp_node, "_out", builder.add_access(block1, "y"), {}, desc);

    auto& block2 = builder.add_block(root);
    auto& log_node = static_cast<math::cmath::CMathNode&>(builder.add_library_node<math::cmath::CMathNode>(
        block2, DebugInfo(), math::cmath::CMathFunction::log, types::PrimitiveType::Float
    ));
    builder.add_computational_memlet(block2, builder.add_access(block2, "y"), log_node, "_in1", {}, desc);
    builder.add_computational_memlet(block2, log_node, "_out", builder.add_access(block2, "z"), {}, desc);

    EXPECT_FALSE(exp_node.approximate());
    EXPECT_FALSE(log_node.approximate());

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ApproximateMathPass pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));
    EXPECT_TRUE(exp_node.approximate());
    EXPECT_TRUE(log_node.approximate());

    // Idempotent: a second run changes nothing.
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}
