#include "sdfg/symbolic/sets.h"

#include <gtest/gtest.h>

#include "sdfg/symbolic/utils.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(SetsTest, is_subset_1d_equivalent) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(1));

    EXPECT_TRUE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_subset_1d_not_equivalent) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(2));

    EXPECT_FALSE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_subset_1d_equivalent_rename) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.add_lower_bound(y);
    assum_x.add_upper_bound(y);

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.add_lower_bound(x);
    assum_y.add_upper_bound(x);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(y, symbolic::integer(1));

    EXPECT_TRUE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_subset_1d_equivalent_recursive_assumptions) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto z = symbolic::symbol("z");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.add_lower_bound(x);
    assum_x.add_upper_bound(z);

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.add_lower_bound(z);
    assum_y.add_upper_bound(y);

    auto assum_z = symbolic::Assumption::create(z, desc);
    assum_z.add_lower_bound(y);
    assum_z.add_upper_bound(x);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});
    assums.insert({z, assum_z});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(y, symbolic::integer(1));

    EXPECT_TRUE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_subset_1d_minmax_assumptions) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    auto K = symbolic::symbol("K");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    assum_x.add_lower_bound(N);
    assum_x.add_lower_bound(M);
    assum_x.add_upper_bound(M);
    assum_x.add_upper_bound(K);

    auto assum_y = symbolic::Assumption::create(y, desc);
    assum_y.add_lower_bound(N);
    assum_y.add_lower_bound(M);
    assum_y.add_upper_bound(M);
    assum_y.add_upper_bound(K);

    symbolic::Assumptions assums;
    assums.insert({x, assum_x});
    assums.insert({y, assum_y});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(y, symbolic::integer(1));

    EXPECT_TRUE(symbolic::is_subset({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_disjoint_1d_disjoint) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(1));

    EXPECT_FALSE(symbolic::is_disjoint({expr1}, {expr2}, assums, assums));
}

TEST(SetsTest, is_disjoint_1d_not_disjoint) {
    auto x = symbolic::symbol("x");
    types::Scalar desc(types::PrimitiveType::UInt8);

    auto assum_x = symbolic::Assumption::create(x, desc);
    symbolic::Assumptions assums;
    assums.insert({x, assum_x});

    auto expr1 = symbolic::add(x, symbolic::integer(1));
    auto expr2 = symbolic::add(x, symbolic::integer(2));

    EXPECT_FALSE(symbolic::is_disjoint({expr1}, {expr2}, assums, assums));
}
