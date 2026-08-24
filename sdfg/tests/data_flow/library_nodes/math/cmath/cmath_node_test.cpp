#include "gtest/gtest.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"

using namespace sdfg;

void TestCMathNodePositive(math::cmath::CMathFunction function, types::PrimitiveType primitive_type) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    types::Scalar desc(primitive_type);
    auto& node =
        static_cast<math::cmath::CMathNode&>(builder.add_library_node<
                                             math::cmath::CMathNode>(block, DebugInfo(), function, primitive_type));

    for (size_t i = 1; i < math::cmath::cmath_function_to_arity(function) + 1; i++) {
        builder.add_container("I_" + std::to_string(i), desc);
        auto& in_node = builder.add_access(block, "I_" + std::to_string(i));
        builder.add_computational_memlet(block, in_node, node, "_in" + std::to_string(i), {}, desc);
    }
    builder.add_container("O_0", desc);
    auto& out_node = builder.add_access(block, "O_0");
    builder.add_computational_memlet(block, node, "_out", out_node, {}, desc);

    EXPECT_NO_THROW(node.validate(sdfg));
}

void TestCMathNodeNegative(math::cmath::CMathFunction function, types::PrimitiveType primitive_type) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();

    auto& block = builder.add_block(sdfg.root());

    types::Scalar desc(primitive_type);
    auto& node =
        static_cast<math::cmath::CMathNode&>(builder.add_library_node<
                                             math::cmath::CMathNode>(block, DebugInfo(), function, primitive_type));

    for (size_t i = 1; i < math::cmath::cmath_function_to_arity(function) + 1; i++) {
        builder.add_container("I_" + std::to_string(i), desc);
        auto& in_node = builder.add_access(block, "I_" + std::to_string(i));
        builder.add_computational_memlet(block, in_node, node, "_in" + std::to_string(i), {}, desc);
    }
    builder.add_container("O_0", desc);
    auto& out_node = builder.add_access(block, "O_0");
    builder.add_computational_memlet(block, node, "_out", out_node, {}, desc);

    EXPECT_THROW(node.validate(sdfg), InvalidSDFGException);
}

#define REGISTER_POSITIVE_TEST(Function, PType)                                                   \
    TEST(CMathNodeTest, Function##_##PType) {                                                     \
        TestCMathNodePositive(math::cmath::CMathFunction::Function, types::PrimitiveType::PType); \
    }

#define REGISTER_NEGATIVE_TEST(Function, PType)                                                   \
    TEST(CMathNodeTest, Function##_##PType##_Negative) {                                          \
        TestCMathNodeNegative(math::cmath::CMathFunction::Function, types::PrimitiveType::PType); \
    }

// Positive Tests
REGISTER_POSITIVE_TEST(acos, Float)
REGISTER_POSITIVE_TEST(sin, Float)
REGISTER_POSITIVE_TEST(cos, Float)
REGISTER_POSITIVE_TEST(tan, Float)
REGISTER_POSITIVE_TEST(asin, Float)
REGISTER_POSITIVE_TEST(atan, Float)
REGISTER_POSITIVE_TEST(atan2, Float)
REGISTER_POSITIVE_TEST(sinh, Float)
REGISTER_POSITIVE_TEST(cosh, Float)
REGISTER_POSITIVE_TEST(tanh, Float)
REGISTER_POSITIVE_TEST(asinh, Float)
REGISTER_POSITIVE_TEST(acosh, Float)
REGISTER_POSITIVE_TEST(atanh, Float)
REGISTER_POSITIVE_TEST(exp, Float)
REGISTER_POSITIVE_TEST(exp2, Float)
REGISTER_POSITIVE_TEST(exp10, Float)
REGISTER_POSITIVE_TEST(expm1, Float)
REGISTER_POSITIVE_TEST(log, Float)
REGISTER_POSITIVE_TEST(log10, Float)
REGISTER_POSITIVE_TEST(log2, Float)
REGISTER_POSITIVE_TEST(log1p, Float)
REGISTER_POSITIVE_TEST(pow, Float)
REGISTER_POSITIVE_TEST(sqrt, Float)
REGISTER_POSITIVE_TEST(cbrt, Float)
REGISTER_POSITIVE_TEST(hypot, Float)
REGISTER_POSITIVE_TEST(erf, Float)
REGISTER_POSITIVE_TEST(erfc, Float)
REGISTER_POSITIVE_TEST(tgamma, Float)
REGISTER_POSITIVE_TEST(lgamma, Float)
REGISTER_POSITIVE_TEST(fabs, Float)
REGISTER_POSITIVE_TEST(ceil, Float)
REGISTER_POSITIVE_TEST(floor, Float)
REGISTER_POSITIVE_TEST(trunc, Float)
REGISTER_POSITIVE_TEST(round, Float)
// REGISTER_POSITIVE_TEST(lround, Float)
// REGISTER_POSITIVE_TEST(llround, Float)
REGISTER_POSITIVE_TEST(nearbyint, Float)
REGISTER_POSITIVE_TEST(rint, Float)
// REGISTER_POSITIVE_TEST(lrint, Float)
// REGISTER_POSITIVE_TEST(llrint, Float)
REGISTER_POSITIVE_TEST(fmod, Float)
REGISTER_POSITIVE_TEST(remainder, Float)
REGISTER_POSITIVE_TEST(frexp, Float)
REGISTER_POSITIVE_TEST(ldexp, Float)
REGISTER_POSITIVE_TEST(modf, Float)
REGISTER_POSITIVE_TEST(scalbn, Float)
REGISTER_POSITIVE_TEST(scalbln, Float)
REGISTER_POSITIVE_TEST(ilogb, Float)
REGISTER_POSITIVE_TEST(logb, Float)
REGISTER_POSITIVE_TEST(nextafter, Float)
REGISTER_POSITIVE_TEST(nexttoward, Float)
REGISTER_POSITIVE_TEST(copysign, Float)
REGISTER_POSITIVE_TEST(fmax, Float)
REGISTER_POSITIVE_TEST(fmin, Float)
REGISTER_POSITIVE_TEST(fdim, Float)

REGISTER_POSITIVE_TEST(acos, Double)
REGISTER_NEGATIVE_TEST(acos, Int32)

// Every cmath input is a read-only value read that never captures the pointer,
// so transforms (e.g. LocalStorage) can localize a container a cmath reads.
TEST(CMathNodeTest, PointerAccessType_ReadOnlyNoCapture) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());
    auto& node = static_cast<math::cmath::CMathNode&>(builder.add_library_node<math::cmath::CMathNode>(
        block, DebugInfo(), math::cmath::CMathFunction::fmax, types::PrimitiveType::Float
    ));

    const int arity = static_cast<int>(math::cmath::cmath_function_to_arity(math::cmath::CMathFunction::fmax));
    ASSERT_EQ(arity, 2);
    for (int i = 0; i < arity; i++) {
        auto access = node.pointer_access_type(i);
        ASSERT_NE(access, nullptr);
        EXPECT_TRUE(access->no_capture());
        EXPECT_TRUE(access->may_contain_reads());
        EXPECT_FALSE(access->may_contain_writes());
    }
    // Out-of-range indices fall back to the base (no metadata).
    EXPECT_EQ(node.pointer_access_type(arity), nullptr);
}
