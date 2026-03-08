#include "sdfg/symbolic/sets.h"

#include <gtest/gtest.h>

#include "sdfg/symbolic/utils.h"
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

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 2);
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(0), x));
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(1), y));
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
    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 3);
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(0), x));
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(1), y));
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(2), z));
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

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 4);
    EXPECT_TRUE(symbolic::eq(expr_delinearized[0], _13));
    EXPECT_TRUE(symbolic::eq(expr_delinearized[1], _24));
    EXPECT_TRUE(symbolic::eq(expr_delinearized[2], _28));
    EXPECT_TRUE(symbolic::eq(expr_delinearized[3], _32));
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

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 4);
    EXPECT_TRUE(symbolic::eq(expr_delinearized[0], _13));
    EXPECT_TRUE(symbolic::eq(expr_delinearized[1], symbolic::add(_24, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(expr_delinearized[2], symbolic::add(_28, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(expr_delinearized[3], symbolic::add(_32, symbolic::one())));
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

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 4);
    EXPECT_TRUE(symbolic::eq(expr_delinearized[0], symbolic::add(symbolic::sub(_13, _4), symbolic::mul(_4, _5))));
    EXPECT_TRUE(symbolic::eq(expr_delinearized[1], symbolic::add(_24, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(expr_delinearized[2], symbolic::add(_28, symbolic::one())));
    EXPECT_TRUE(symbolic::eq(expr_delinearized[3], symbolic::add(_32, symbolic::one())));
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

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 1);
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(0), expr));
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

    auto expr_delinearized = symbolic::delinearize({expr}, assums);
    EXPECT_EQ(expr_delinearized.size(), 1);
    EXPECT_TRUE(symbolic::eq(expr_delinearized.at(0), expr));
}
