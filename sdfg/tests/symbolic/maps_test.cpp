#include "sdfg/symbolic/sets.h"

#include <gtest/gtest.h>

#include "sdfg/symbolic/maps.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(MapsTest, IsDisjoint_Stencil7P) {
    types::Scalar desc_i64(types::PrimitiveType::Int32);
    types::Scalar desc_i32(types::PrimitiveType::Int32);

    // Bounds
    auto _19 = symbolic::symbol("_19");
    auto assum_19 = symbolic::Assumption::create(_19, desc_i32);
    assum_19.add_lower_bound(symbolic::one());
    assum_19.constant(true);

    auto _1 = symbolic::symbol("_1");
    auto assum_1 = symbolic::Assumption::create(_1, desc_i64);
    assum_1.add_lower_bound(symbolic::one());
    assum_1.constant(true);

    auto _2 = symbolic::symbol("_2");
    auto assum_2 = symbolic::Assumption::create(_2, desc_i64);
    assum_2.add_lower_bound(symbolic::one());
    assum_2.constant(true);

    auto _3 = symbolic::symbol("_3");
    auto assum_3 = symbolic::Assumption::create(_3, desc_i64);
    assum_3.add_lower_bound(symbolic::one());
    assum_3.constant(true);

    // Indvars
    auto _13 = symbolic::symbol("_13");
    auto assum_13 = symbolic::Assumption::create(_13, desc_i64);
    assum_13.add_lower_bound(symbolic::zero());
    assum_13.add_upper_bound(symbolic::sub(_19, symbolic::one()));
    assum_13.map(symbolic::add(_13, symbolic::one()));

    auto _24 = symbolic::symbol("_24");
    auto assum_24 = symbolic::Assumption::create(_24, desc_i64);
    assum_24.add_lower_bound(symbolic::zero());
    assum_24.add_upper_bound(symbolic::sub(_1, symbolic::one()));
    assum_24.map(symbolic::add(_24, symbolic::one()));

    auto _28 = symbolic::symbol("_28");
    auto assum_28 = symbolic::Assumption::create(_28, desc_i64);
    assum_28.add_lower_bound(symbolic::zero());
    assum_28.add_upper_bound(symbolic::sub(_2, symbolic::one()));
    assum_28.map(symbolic::add(_28, symbolic::one()));

    auto _32 = symbolic::symbol("_32");
    auto assum_32 = symbolic::Assumption::create(_32, desc_i64);
    assum_32.add_lower_bound(symbolic::zero());
    assum_32.add_upper_bound(symbolic::sub(_3, symbolic::one()));
    assum_32.map(symbolic::add(_32, symbolic::one()));

    symbolic::Assumptions assums1;
    assums1.insert({_13, assum_13});
    assums1.insert({_24, assum_24});
    assums1.insert({_28, assum_28});
    assums1.insert({_32, assum_32});
    assums1.insert({_19, assum_19});
    assums1.insert({_1, assum_1});
    assums1.insert({_2, assum_2});
    assums1.insert({_3, assum_3});

    auto assums2 = assums1;

    // 1 + _32 + (2 + _3)*(1 + _28) + (2 + _3)*(2 + _2)*(1 + _24) + (2 + _3)*(2 + _2)*(2 + _1)*_13
    auto stride_1 = symbolic::add(symbolic::integer(2), _1);
    auto stride_2 = symbolic::add(symbolic::integer(2), _2);
    auto stride_3 = symbolic::add(symbolic::integer(2), _3);
    auto offset_32 = symbolic::add(symbolic::one(), _32);
    auto offset_28 = symbolic::add(symbolic::one(), _28);
    auto offset_24 = symbolic::add(symbolic::one(), _24);
    auto expr1 = symbolic::add(offset_32, symbolic::mul(stride_3, offset_28));
    expr1 = symbolic::add(expr1, symbolic::mul(symbolic::mul(stride_3, stride_2), offset_24));
    expr1 = symbolic::add(expr1, symbolic::mul(symbolic::mul(symbolic::mul(stride_3, stride_2), stride_1), _13));

    // 1 + _32 + (2 + _3)*(1 + _28) + (2 + _3)*(2 + _2)*_24 + (2 + _3)*(2 + _2)*(2 + _1)*_13
    auto expr2 = symbolic::add(offset_32, symbolic::mul(stride_3, offset_28));
    expr2 = symbolic::add(expr2, symbolic::mul(symbolic::mul(stride_3, stride_2), _24));
    expr2 = symbolic::add(expr2, symbolic::mul(symbolic::mul(symbolic::mul(stride_3, stride_2), stride_1), _13));
    EXPECT_FALSE(symbolic::maps::intersects({expr1}, {expr2}, _13, assums1, assums2));
    EXPECT_TRUE(symbolic::maps::intersects({expr1}, {expr2}, _24, assums1, assums2));

    // 1 + _32 + (2 + _3)*_28 + (2 + _3)*(2 + _2)*(1 + _24) + (2 + _3)*(2 + _2)*(2 + _1)*_13
    auto expr3 = symbolic::add(offset_32, symbolic::mul(stride_3, _28));
    expr3 = symbolic::add(expr3, symbolic::mul(symbolic::mul(stride_3, stride_2), offset_24));
    expr3 = symbolic::add(expr3, symbolic::mul(symbolic::mul(symbolic::mul(stride_3, stride_2), stride_1), _13));
    EXPECT_FALSE(symbolic::maps::intersects({expr1}, {expr3}, _13, assums1, assums2));
    EXPECT_TRUE(symbolic::maps::intersects({expr1}, {expr3}, _28, assums1, assums2));

    // _32 + (2 + _3)*(1 + _28) + (2 + _3)*(2 + _2)*(1 + _24) + (2 + _3)*(2 + _2)*(2 + _1)*_13
    auto expr4 = symbolic::add(_32, symbolic::mul(stride_3, offset_28));
    expr4 = symbolic::add(expr4, symbolic::mul(symbolic::mul(stride_3, stride_2), offset_24));
    expr4 = symbolic::add(expr4, symbolic::mul(symbolic::mul(symbolic::mul(stride_3, stride_2), stride_1), _13));
    EXPECT_FALSE(symbolic::maps::intersects({expr1}, {expr4}, _13, assums1, assums2));
    EXPECT_TRUE(symbolic::maps::intersects({expr1}, {expr4}, _32, assums1, assums2));
}
