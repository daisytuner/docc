#include <gtest/gtest.h>

#include "sdfg/builder/sdfg_builder.h"
using namespace sdfg;

TEST(TaskletTest, Casts_Trivial) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt8);
    types::Scalar desc2(types::PrimitiveType::UInt8);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_trivial(builder.subject()));
    EXPECT_FALSE(tasklet_1.is_cast(builder.subject()));
}

TEST(TaskletTest, Casts_Zext) {
    builder::SDFGBuilder builder("sdfg_1", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt8);
    types::Scalar desc2(types::PrimitiveType::UInt32);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_zext(builder.subject()));
}

TEST(TaskletTest, Casts_Sext) {
    builder::SDFGBuilder builder("sdfg_2", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int8);
    types::Scalar desc2(types::PrimitiveType::Int32);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_sext(builder.subject()));
}

TEST(TaskletTest, Casts_Trunc) {
    builder::SDFGBuilder builder("sdfg_3", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Scalar desc2(types::PrimitiveType::UInt8);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_trunc(builder.subject()));
}

TEST(TaskletTest, Casts_Fptoui) {
    builder::SDFGBuilder builder("sdfg_4", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Scalar desc2(types::PrimitiveType::UInt32);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_fptoui(builder.subject()));
}

TEST(TaskletTest, Casts_Fptosi) {
    builder::SDFGBuilder builder("sdfg_5", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Scalar desc2(types::PrimitiveType::Int32);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_fptosi(builder.subject()));
}

TEST(TaskletTest, Casts_Uitofp) {
    builder::SDFGBuilder builder("sdfg_6", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::UInt32);
    types::Scalar desc2(types::PrimitiveType::Float);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_uitofp(builder.subject()));
}

TEST(TaskletTest, Casts_Sitofp) {
    builder::SDFGBuilder builder("sdfg_7", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Int32);
    types::Scalar desc2(types::PrimitiveType::Float);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_sitofp(builder.subject()));
}

TEST(TaskletTest, Casts_Fpext) {
    builder::SDFGBuilder builder("sdfg_8", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Scalar desc2(types::PrimitiveType::Double);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_fpext(builder.subject()));
}

TEST(TaskletTest, Casts_Fptrunc) {
    builder::SDFGBuilder builder("sdfg_9", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Scalar desc2(types::PrimitiveType::Float);
    builder.add_container("i1", desc);
    builder.add_container("i2", desc2);

    auto& access_node_1 = builder.add_access(state, "i1");
    auto& access_node_2 = builder.add_access(state, "i2");
    auto& tasklet_1 = builder.add_tasklet(state, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(state, access_node_1, tasklet_1, "_in", {});
    builder.add_computational_memlet(state, tasklet_1, "_out", access_node_2, {});

    EXPECT_TRUE(tasklet_1.is_assign());
    EXPECT_TRUE(tasklet_1.is_cast(builder.subject()));
    EXPECT_TRUE(tasklet_1.is_fptrunc(builder.subject()));
}

TEST(TaskletTest, Complex_Arity) {
    EXPECT_EQ(data_flow::arity(data_flow::TaskletCode::complex_real), 1);
    EXPECT_EQ(data_flow::arity(data_flow::TaskletCode::complex_imag), 1);
    EXPECT_EQ(data_flow::arity(data_flow::TaskletCode::complex_neg), 1);
    EXPECT_EQ(data_flow::arity(data_flow::TaskletCode::complex_add), 2);
    EXPECT_EQ(data_flow::arity(data_flow::TaskletCode::complex_sub), 2);
    EXPECT_EQ(data_flow::arity(data_flow::TaskletCode::complex_mul), 2);
    EXPECT_EQ(data_flow::arity(data_flow::TaskletCode::complex_div), 2);
    EXPECT_EQ(data_flow::arity(data_flow::TaskletCode::complex_eq), 2);
    EXPECT_EQ(data_flow::arity(data_flow::TaskletCode::complex_ne), 2);
}

TEST(TaskletTest, Complex_Classification) {
    const std::vector<data_flow::TaskletCode> complex_codes = {
        data_flow::TaskletCode::complex_real,
        data_flow::TaskletCode::complex_imag,
        data_flow::TaskletCode::complex_neg,
        data_flow::TaskletCode::complex_add,
        data_flow::TaskletCode::complex_sub,
        data_flow::TaskletCode::complex_mul,
        data_flow::TaskletCode::complex_div,
        data_flow::TaskletCode::complex_eq,
        data_flow::TaskletCode::complex_ne,
    };

    for (auto code : complex_codes) {
        EXPECT_TRUE(data_flow::is_complex(code));
        EXPECT_TRUE(data_flow::is_floating_point(code));
        EXPECT_FALSE(data_flow::is_integer(code));
        EXPECT_FALSE(data_flow::is_unsigned(code));
    }

    // Non-complex operations must not be classified as complex.
    EXPECT_FALSE(data_flow::is_complex(data_flow::TaskletCode::assign));
    EXPECT_FALSE(data_flow::is_complex(data_flow::TaskletCode::fp_add));
    EXPECT_FALSE(data_flow::is_complex(data_flow::TaskletCode::int_add));
}

TEST(TaskletTest, Complex_Add_Validate) {
    builder::SDFGBuilder builder("sdfg_complex_add", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::CFloat);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& c_node = builder.add_access(state, "c");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::complex_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, a_node, tasklet, "_in1", {});
    builder.add_computational_memlet(state, b_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", c_node, {});

    EXPECT_EQ(tasklet.code(), data_flow::TaskletCode::complex_add);
    EXPECT_NO_THROW(tasklet.validate(builder.subject()));
}

TEST(TaskletTest, Complex_Mul_Validate) {
    builder::SDFGBuilder builder("sdfg_complex_mul", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::CDouble);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& c_node = builder.add_access(state, "c");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::complex_mul, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, a_node, tasklet, "_in1", {});
    builder.add_computational_memlet(state, b_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", c_node, {});

    EXPECT_NO_THROW(tasklet.validate(builder.subject()));
}

TEST(TaskletTest, Complex_Neg_Validate) {
    builder::SDFGBuilder builder("sdfg_complex_neg", FunctionType_CPU);

    auto& state = builder.add_state();

    types::Scalar desc(types::PrimitiveType::CFloat);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::complex_neg, "_out", {"_in"});
    builder.add_computational_memlet(state, a_node, tasklet, "_in", {});
    builder.add_computational_memlet(state, tasklet, "_out", b_node, {});

    EXPECT_NO_THROW(tasklet.validate(builder.subject()));
}

TEST(TaskletTest, Complex_Validate_RejectsNonComplexInput) {
    builder::SDFGBuilder builder("sdfg_complex_invalid", FunctionType_CPU);

    auto& state = builder.add_state();

    // Feed a real floating-point input into a complex operation.
    types::Scalar desc(types::PrimitiveType::Float);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto& a_node = builder.add_access(state, "a");
    auto& b_node = builder.add_access(state, "b");
    auto& c_node = builder.add_access(state, "c");
    auto& tasklet = builder.add_tasklet(state, data_flow::TaskletCode::complex_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(state, a_node, tasklet, "_in1", {});
    builder.add_computational_memlet(state, b_node, tasklet, "_in2", {});
    builder.add_computational_memlet(state, tasklet, "_out", c_node, {});

    EXPECT_THROW(tasklet.validate(builder.subject()), InvalidSDFGException);
}
