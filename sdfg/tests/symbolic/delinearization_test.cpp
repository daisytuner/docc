#include "sdfg/symbolic/sets.h"

#include <gtest/gtest.h>

#include "sdfg/symbolic/delinearization.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(DelinearizeTest, Delinearize2D) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound_deprecated(symbolic::zero());
    assum_x.upper_bound_deprecated(symbolic::sub(N, symbolic::integer(1)));

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound_deprecated(symbolic::zero());
    assum_y.upper_bound_deprecated(symbolic::sub(M, symbolic::integer(1)));

    auto assum_N = symbolic::Assumption::create(N, desc);
    assum_N.lower_bound_deprecated(symbolic::integer(1));
    assum_N.upper_bound_deprecated(symbolic::integer(10));
    assum_N.constant(true);

    auto assum_M = symbolic::Assumption::create(M, desc);
    assum_M.lower_bound_deprecated(symbolic::integer(1));
    assum_M.upper_bound_deprecated(symbolic::integer(20));
    assum_M.constant(true);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({N, assum_N});
    assums.insert({M, assum_M});

    auto expr = symbolic::add(symbolic::mul(x, M), y);

    auto result = symbolic::delinearize(expr, assums);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.dimensions.size(), 1);
    EXPECT_EQ(result.indices.size(), 2);
    EXPECT_TRUE(symbolic::eq(result.indices.at(0), x));
    EXPECT_TRUE(symbolic::eq(result.indices.at(1), y));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(0), M));
}

TEST(DelinearizeTest, Delinearize3D) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto z = symbolic::symbol("z");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound_deprecated(symbolic::zero());
    assum_x.upper_bound_deprecated(symbolic::sub(N, symbolic::integer(1)));

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound_deprecated(symbolic::zero());
    assum_y.upper_bound_deprecated(symbolic::sub(M, symbolic::integer(1)));

    auto assum_z = symbolic::Assumption::create(z, desc);
    assum_z.lower_bound_deprecated(symbolic::zero());
    assum_z.upper_bound_deprecated(symbolic::sub(K, symbolic::integer(1)));

    auto assum_N = symbolic::Assumption::create(N, desc);
    assum_N.lower_bound_deprecated(symbolic::integer(1));
    assum_N.upper_bound_deprecated(symbolic::integer(10));
    assum_N.constant(true);

    auto assum_M = symbolic::Assumption::create(M, desc);
    assum_M.lower_bound_deprecated(symbolic::integer(1));
    assum_M.upper_bound_deprecated(symbolic::integer(20));
    assum_M.constant(true);

    auto assum_K = symbolic::Assumption::create(K, desc);
    assum_K.lower_bound_deprecated(symbolic::integer(1));
    assum_K.upper_bound_deprecated(symbolic::integer(30));
    assum_K.constant(true);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({z, assum_z});
    assums.insert({N, assum_N});
    assums.insert({M, assum_M});
    assums.insert({K, assum_K});

    auto expr = symbolic::add(symbolic::add(symbolic::mul(x, symbolic::mul(M, K)), symbolic::mul(y, K)), z);
    auto result = symbolic::delinearize(expr, assums);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.dimensions.size(), 2);
    EXPECT_EQ(result.indices.size(), 3);
    for (auto& dim : result.dimensions) {
        std::cout << dim->__str__() << std::endl;
    }
    for (auto& idx : result.indices) {
        std::cout << idx->__str__() << std::endl;
    }

    EXPECT_TRUE(symbolic::eq(result.indices.at(0), x));
    EXPECT_TRUE(symbolic::eq(result.indices.at(1), y));
    EXPECT_TRUE(symbolic::eq(result.indices.at(2), z));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(0), M));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(1), K));
}


TEST(DelinearizeTest, Delinearize4D) {
    types::Scalar desc_i64(types::PrimitiveType::Int64);
    types::Scalar desc_i32(types::PrimitiveType::Int32);

    // Bounds
    auto _19 = symbolic::symbol("_19");
    auto assums_19 = symbolic::Assumption::create(_19, desc_i32);
    assums_19.lower_bound_deprecated(symbolic::integer(1));
    assums_19.constant(true);

    auto _1 = symbolic::symbol("_1");
    auto assums_1 = symbolic::Assumption::create(_1, desc_i64);
    assums_1.lower_bound_deprecated(symbolic::integer(1));
    assums_1.constant(true);

    auto _2 = symbolic::symbol("_2");
    auto assums_2 = symbolic::Assumption::create(_2, desc_i64);
    assums_2.lower_bound_deprecated(symbolic::integer(1));
    assums_2.constant(true);

    auto _3 = symbolic::symbol("_3");
    auto assums_3 = symbolic::Assumption::create(_3, desc_i64);
    assums_3.lower_bound_deprecated(symbolic::integer(1));
    assums_3.constant(true);

    // Indvars
    auto _13 = symbolic::symbol("_13");
    auto assum_13 = symbolic::Assumption::create(_13, desc_i64);
    assum_13.lower_bound_deprecated(symbolic::zero());
    assum_13.upper_bound_deprecated(symbolic::sub(_19, symbolic::one()));
    assum_13.map(symbolic::add(_13, symbolic::one()));

    auto _24 = symbolic::symbol("_24");
    auto assum_24 = symbolic::Assumption::create(_24, desc_i64);
    assum_24.lower_bound_deprecated(symbolic::zero());
    assum_24.upper_bound_deprecated(symbolic::sub(_1, symbolic::one()));
    assum_24.map(symbolic::add(_24, symbolic::one()));

    auto _28 = symbolic::symbol("_28");
    auto assum_28 = symbolic::Assumption::create(_28, desc_i64);
    assum_28.lower_bound_deprecated(symbolic::zero());
    assum_28.upper_bound_deprecated(symbolic::sub(_2, symbolic::one()));
    assum_28.map(symbolic::add(_28, symbolic::one()));

    auto _32 = symbolic::symbol("_32");
    auto assum_32 = symbolic::Assumption::create(_32, desc_i64);
    assum_32.lower_bound_deprecated(symbolic::zero());
    assum_32.upper_bound_deprecated(symbolic::sub(_3, symbolic::one()));
    assum_32.map(symbolic::add(_32, symbolic::one()));

    symbolic::Assumptions assums;
    assums.insert({_13, assum_13});
    assums.insert({_24, assum_24});
    assums.insert({_28, assum_28});
    assums.insert({_32, assum_32});
    assums.insert({_19, assums_19});
    assums.insert({_1, assums_1});
    assums.insert({_2, assums_2});
    assums.insert({_3, assums_3});

    auto expr = symbolic::add(_32, symbolic::mul(_3, _28));
    expr = symbolic::add(expr, symbolic::mul(symbolic::mul(_3, _2), _24));
    expr = symbolic::add(expr, symbolic::mul(symbolic::mul(symbolic::mul(_3, _2), _1), _13));

    auto result = symbolic::delinearize(expr, assums);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.dimensions.size(), 3);
    EXPECT_EQ(result.indices.size(), 4);
    EXPECT_TRUE(symbolic::eq(result.indices.at(0), _13));
    EXPECT_TRUE(symbolic::eq(result.indices.at(1), _24));
    EXPECT_TRUE(symbolic::eq(result.indices.at(2), _28));
    EXPECT_TRUE(symbolic::eq(result.indices.at(3), _32));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(0), _1));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(1), _2));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(2), _3));
}

TEST(DelinearizeTest, Delinearize4D_WithOffsets) {
    types::Scalar desc_i64(types::PrimitiveType::Int64);
    types::Scalar desc_i32(types::PrimitiveType::Int32);

    // Bounds
    auto _19 = symbolic::symbol("_19");
    auto assums_19 = symbolic::Assumption::create(_19, desc_i32);
    assums_19.lower_bound_deprecated(symbolic::integer(1));
    assums_19.constant(true);

    auto _1 = symbolic::symbol("_1");
    auto assums_1 = symbolic::Assumption::create(_1, desc_i64);
    assums_1.lower_bound_deprecated(symbolic::integer(1));
    assums_1.constant(true);

    auto _2 = symbolic::symbol("_2");
    auto assums_2 = symbolic::Assumption::create(_2, desc_i64);
    assums_2.lower_bound_deprecated(symbolic::integer(1));
    assums_2.constant(true);

    auto _3 = symbolic::symbol("_3");
    auto assums_3 = symbolic::Assumption::create(_3, desc_i64);
    assums_3.lower_bound_deprecated(symbolic::integer(1));
    assums_3.constant(true);

    // Indvars
    auto _13 = symbolic::symbol("_13");
    auto assum_13 = symbolic::Assumption::create(_13, desc_i64);
    assum_13.lower_bound_deprecated(symbolic::zero());
    assum_13.upper_bound_deprecated(symbolic::sub(_19, symbolic::one()));
    assum_13.map(symbolic::add(_13, symbolic::one()));

    auto _24 = symbolic::symbol("_24");
    auto assum_24 = symbolic::Assumption::create(_24, desc_i64);
    assum_24.lower_bound_deprecated(symbolic::zero());
    assum_24.upper_bound_deprecated(symbolic::sub(_1, symbolic::one()));
    assum_24.map(symbolic::add(_24, symbolic::one()));

    auto _28 = symbolic::symbol("_28");
    auto assum_28 = symbolic::Assumption::create(_28, desc_i64);
    assum_28.lower_bound_deprecated(symbolic::zero());
    assum_28.upper_bound_deprecated(symbolic::sub(_2, symbolic::one()));
    assum_28.map(symbolic::add(_28, symbolic::one()));

    auto _32 = symbolic::symbol("_32");
    auto assum_32 = symbolic::Assumption::create(_32, desc_i64);
    assum_32.lower_bound_deprecated(symbolic::zero());
    assum_32.upper_bound_deprecated(symbolic::sub(_3, symbolic::one()));
    assum_32.map(symbolic::add(_32, symbolic::one()));

    symbolic::Assumptions assums;
    assums.insert({_13, assum_13});
    assums.insert({_24, assum_24});
    assums.insert({_28, assum_28});
    assums.insert({_32, assum_32});
    assums.insert({_19, assums_19});
    assums.insert({_1, assums_1});
    assums.insert({_2, assums_2});
    assums.insert({_3, assums_3});

    // 1 + _32 + (2 + _3)*(1 + _28) + (2 + _3)*(2 + _2)*(1 + _24) + (2 + _3)*(2 + _2)*(2 + _1)*_13
    auto stride_1 = symbolic::add(symbolic::integer(2), _1);
    auto stride_2 = symbolic::add(symbolic::integer(2), _2);
    auto stride_3 = symbolic::add(symbolic::integer(2), _3);
    auto offset_32 = symbolic::add(symbolic::one(), _32);
    auto offset_28 = symbolic::add(symbolic::one(), _28);
    auto offset_24 = symbolic::add(symbolic::one(), _24);
    auto expr = symbolic::add(offset_32, symbolic::mul(stride_3, offset_28));
    expr = symbolic::add(expr, symbolic::mul(symbolic::mul(stride_3, stride_2), offset_24));
    expr = symbolic::add(expr, symbolic::mul(symbolic::mul(symbolic::mul(stride_3, stride_2), stride_1), _13));

    auto result = symbolic::delinearize(expr, assums);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.dimensions.size(), 3);
    EXPECT_EQ(result.indices.size(), 4);
    EXPECT_TRUE(symbolic::eq(result.indices.at(0), _13));
    EXPECT_TRUE(symbolic::eq(result.indices.at(1), offset_24));
    EXPECT_TRUE(symbolic::eq(result.indices.at(2), offset_28));
    EXPECT_TRUE(symbolic::eq(result.indices.at(3), offset_32));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(0), stride_1));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(1), stride_2));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(2), stride_3));
}

TEST(DelinearizeTest, Delinearize4D_WithOffsets_Symbolic) {
    types::Scalar desc_i64(types::PrimitiveType::Int64);
    types::Scalar desc_i32(types::PrimitiveType::Int32);

    // Bounds
    auto _19 = symbolic::symbol("_19");
    auto assums_19 = symbolic::Assumption::create(_19, desc_i32);
    assums_19.lower_bound_deprecated(symbolic::integer(1));
    assums_19.constant(true);

    auto _1 = symbolic::symbol("_1");
    auto assums_1 = symbolic::Assumption::create(_1, desc_i64);
    assums_1.lower_bound_deprecated(symbolic::integer(1));
    assums_1.constant(true);

    auto _2 = symbolic::symbol("_2");
    auto assums_2 = symbolic::Assumption::create(_2, desc_i64);
    assums_2.lower_bound_deprecated(symbolic::integer(1));
    assums_2.constant(true);

    auto _3 = symbolic::symbol("_3");
    auto assums_3 = symbolic::Assumption::create(_3, desc_i64);
    assums_3.lower_bound_deprecated(symbolic::integer(1));
    assums_3.constant(true);

    auto _4 = symbolic::symbol("_4");
    auto assums_4 = symbolic::Assumption::create(_4, desc_i64);
    assums_4.constant(true);

    auto _5 = symbolic::symbol("_5");
    auto assums_5 = symbolic::Assumption::create(_5, desc_i64);
    assums_5.constant(true);

    // Indvars
    auto _13 = symbolic::symbol("_13");
    auto assum_13 = symbolic::Assumption::create(_13, desc_i64);
    assum_13.lower_bound_deprecated(symbolic::zero());
    assum_13.upper_bound_deprecated(symbolic::sub(_19, symbolic::one()));
    assum_13.map(symbolic::add(_13, symbolic::one()));

    auto _24 = symbolic::symbol("_24");
    auto assum_24 = symbolic::Assumption::create(_24, desc_i64);
    assum_24.lower_bound_deprecated(symbolic::zero());
    assum_24.upper_bound_deprecated(symbolic::sub(_1, symbolic::one()));
    assum_24.map(symbolic::add(_24, symbolic::one()));

    auto _28 = symbolic::symbol("_28");
    auto assum_28 = symbolic::Assumption::create(_28, desc_i64);
    assum_28.lower_bound_deprecated(symbolic::zero());
    assum_28.upper_bound_deprecated(symbolic::sub(_2, symbolic::one()));
    assum_28.map(symbolic::add(_28, symbolic::one()));

    auto _32 = symbolic::symbol("_32");
    auto assum_32 = symbolic::Assumption::create(_32, desc_i64);
    assum_32.lower_bound_deprecated(symbolic::zero());
    assum_32.upper_bound_deprecated(symbolic::sub(_3, symbolic::one()));
    assum_32.map(symbolic::add(_32, symbolic::one()));

    symbolic::Assumptions assums;
    assums.insert({_13, assum_13});
    assums.insert({_24, assum_24});
    assums.insert({_28, assum_28});
    assums.insert({_32, assum_32});
    assums.insert({_19, assums_19});
    assums.insert({_1, assums_1});
    assums.insert({_2, assums_2});
    assums.insert({_3, assums_3});
    assums.insert({_4, assums_4});
    assums.insert({_5, assums_5});

    // 1 + _32 + (2 + _3)*(1 + _28) + (2 + _3)*(2 + _2)*(1 + _24) + (2 + _3)*(2 + _2)*(2 + _1)*_13 + _4*(_5 - 1)*(2 +
    // _3)*(2 + _2)*(2 + _1)
    auto stride_1 = symbolic::add(symbolic::integer(2), _1);
    auto stride_2 = symbolic::add(symbolic::integer(2), _2);
    auto stride_3 = symbolic::add(symbolic::integer(2), _3);
    auto offset_32 = symbolic::add(symbolic::one(), _32);
    auto offset_28 = symbolic::add(symbolic::one(), _28);
    auto offset_24 = symbolic::add(symbolic::one(), _24);
    auto expr = symbolic::add(offset_32, symbolic::mul(stride_3, offset_28));
    expr = symbolic::add(expr, symbolic::mul(symbolic::mul(stride_3, stride_2), offset_24));
    expr = symbolic::add(expr, symbolic::mul(symbolic::mul(symbolic::mul(stride_3, stride_2), stride_1), _13));
    expr = symbolic::add(
        expr,
        symbolic::
            mul(symbolic::mul(symbolic::mul(symbolic::mul(_4, symbolic::sub(_5, symbolic::one())), stride_3), stride_2),
                stride_1)
    );

    auto result = symbolic::delinearize({expr}, assums);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.dimensions.size(), 3);
    EXPECT_EQ(result.indices.size(), 4);
    EXPECT_TRUE(symbolic::eq(result.indices.at(0), symbolic::add(symbolic::sub(_13, _4), symbolic::mul(_4, _5))));
    EXPECT_TRUE(symbolic::eq(result.indices.at(1), offset_24));
    EXPECT_TRUE(symbolic::eq(result.indices.at(2), offset_28));
    EXPECT_TRUE(symbolic::eq(result.indices.at(3), offset_32));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(0), stride_1));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(1), stride_2));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(2), stride_3));
}

TEST(DelinearizeTest, Delinearize4D_PowStride) {
    // 4D: T[i + s0*i1 + s0*s1*i2 + s0*s1**2*i3]
    using namespace symbolic;
    types::Scalar desc(types::PrimitiveType::Int64);
    auto i0 = symbolic::symbol("i0");
    auto i1 = symbolic::symbol("i1");
    auto i2 = symbolic::symbol("i2");
    auto i3 = symbolic::symbol("i3");
    auto s0 = symbolic::symbol("s0");
    auto s1 = symbolic::symbol("s1");

    symbolic::Assumptions assums;
    auto assum_i0 = symbolic::Assumption::create(i0, desc);
    assum_i0.lower_bound_deprecated(symbolic::zero());
    assum_i0.upper_bound_deprecated(symbolic::sub(s0, symbolic::one()));
    assum_i0.map(symbolic::add(i0, symbolic::one()));
    assum_i0.constant(true);
    assums.insert({i0, assum_i0});

    auto assum_i1 = symbolic::Assumption::create(i1, desc);
    assum_i1.lower_bound_deprecated(symbolic::zero());
    assum_i1.upper_bound_deprecated(symbolic::sub(s1, symbolic::one()));
    assum_i1.map(symbolic::add(i1, symbolic::one()));
    assum_i1.constant(true);
    assums.insert({i1, assum_i1});

    auto assum_i2 = symbolic::Assumption::create(i2, desc);
    assum_i2.lower_bound_deprecated(symbolic::zero());
    assum_i2.upper_bound_deprecated(symbolic::sub(s1, symbolic::one()));
    assum_i2.map(symbolic::add(i2, symbolic::one()));
    assum_i2.constant(true);
    assums.insert({i2, assum_i2});

    auto assum_i3 = symbolic::Assumption::create(i3, desc);
    assum_i3.lower_bound_deprecated(symbolic::zero());
    assum_i3.upper_bound_deprecated(symbolic::sub(s1, symbolic::one()));
    assum_i3.map(symbolic::add(i3, symbolic::one()));
    assum_i3.constant(true);
    assums.insert({i3, assum_i3});

    auto assum_s0 = symbolic::Assumption::create(s0, desc);
    assum_s0.lower_bound_deprecated(symbolic::integer(1));
    assum_s0.constant(true);
    assums.insert({s0, assum_s0});

    auto assum_s1 = symbolic::Assumption::create(s1, desc);
    assum_s1.lower_bound_deprecated(symbolic::integer(1));
    assum_s1.constant(true);
    assums.insert({s1, assum_s1});

    // _i3 + _s1*_i2 + _s1**2*_i1 + _s0*_s1**2*_i0
    auto s1_2 = symbolic::pow(s1, symbolic::integer(2));
    auto expr = symbolic::mul(symbolic::mul(s0, s1_2), i0);
    expr = symbolic::add(expr, symbolic::mul(s1_2, i1));
    expr = symbolic::add(expr, symbolic::mul(s1, i2));
    expr = symbolic::add(expr, i3);

    auto result = symbolic::delinearize(expr, assums);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.dimensions.size(), 3);
    EXPECT_EQ(result.indices.size(), 4);
    EXPECT_TRUE(symbolic::eq(result.indices.at(0), i0));
    EXPECT_TRUE(symbolic::eq(result.indices.at(1), i1));
    EXPECT_TRUE(symbolic::eq(result.indices.at(2), i2));
    EXPECT_TRUE(symbolic::eq(result.indices.at(3), i3));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(0), s0));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(1), s1));
    EXPECT_TRUE(symbolic::eq(result.dimensions.at(2), s1));
}

TEST(DelinearizeTest, ZeroStride) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound_deprecated(symbolic::zero());
    assum_x.upper_bound_deprecated(symbolic::sub(N, symbolic::integer(1)));

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound_deprecated(symbolic::zero());
    assum_y.upper_bound_deprecated(symbolic::sub(M, symbolic::integer(1)));

    auto assum_N = symbolic::Assumption::create(N, desc);
    assum_N.lower_bound_deprecated(symbolic::integer(0));
    assum_N.upper_bound_deprecated(symbolic::integer(10));
    assum_N.constant(true);

    auto assum_M = symbolic::Assumption::create(M, desc);
    assum_M.lower_bound_deprecated(symbolic::integer(0));
    assum_M.upper_bound_deprecated(symbolic::integer(20));
    assum_M.constant(true);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({N, assum_N});
    assums.insert({M, assum_M});

    auto expr = symbolic::add(symbolic::mul(x, M), y);

    auto result = symbolic::delinearize({expr}, assums);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.dimensions.size(), 0);
    EXPECT_EQ(result.indices.size(), 1);
    EXPECT_TRUE(symbolic::eq(result.indices.at(0), expr));
}

TEST(DelinearizeTest, NegativeStride) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.lower_bound_deprecated(symbolic::integer(-1));
    assum_x.upper_bound_deprecated(symbolic::sub(N, symbolic::integer(1)));

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.lower_bound_deprecated(symbolic::integer(-1));
    assum_y.upper_bound_deprecated(symbolic::sub(M, symbolic::integer(1)));

    auto assum_N = symbolic::Assumption::create(N, desc);
    assum_N.lower_bound_deprecated(symbolic::integer(1));
    assum_N.upper_bound_deprecated(symbolic::integer(10));
    assum_N.constant(true);

    auto assum_M = symbolic::Assumption::create(M, desc);
    assum_M.lower_bound_deprecated(symbolic::integer(1));
    assum_M.upper_bound_deprecated(symbolic::integer(20));
    assum_M.constant(true);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({N, assum_N});
    assums.insert({M, assum_M});

    auto expr = symbolic::add(symbolic::mul(x, M), y);

    auto result = symbolic::delinearize({expr}, assums);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.dimensions.size(), 0);
    EXPECT_EQ(result.indices.size(), 1);
    EXPECT_TRUE(symbolic::eq(result.indices.at(0), expr));
}
