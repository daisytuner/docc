#include "sdfg/symbolic/assumptions.h"

#include <gtest/gtest.h>

#include "sdfg/types/scalar.h"

using namespace sdfg;

TEST(AssumptionsTest, Init) {
    auto x = symbolic::symbol("x");

    symbolic::Assumption a(x);
    EXPECT_TRUE(symbolic::eq(a.symbol(), x));
    EXPECT_TRUE(a.lower_bounds().empty());
    EXPECT_TRUE(a.upper_bounds().empty());
    EXPECT_TRUE(a.tight_lower_bound().is_null());
    EXPECT_TRUE(a.tight_upper_bound().is_null());
    EXPECT_FALSE(a.constant());
    EXPECT_TRUE(a.map().is_null());
}

TEST(AssumptionsTest, Create) {
    auto x = symbolic::symbol("x");

    auto a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Bool));
    EXPECT_TRUE(symbolic::eq(*a.lower_bounds().begin(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(*a.upper_bounds().begin(), symbolic::one()));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::UInt8));
    EXPECT_TRUE(symbolic::eq(*a.lower_bounds().begin(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(*a.upper_bounds().begin(), symbolic::integer(std::numeric_limits<uint8_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Int8));
    EXPECT_TRUE(symbolic::eq(*a.lower_bounds().begin(), symbolic::integer(std::numeric_limits<int8_t>::min())));
    EXPECT_TRUE(symbolic::eq(*a.upper_bounds().begin(), symbolic::integer(std::numeric_limits<int8_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::UInt16));
    EXPECT_TRUE(symbolic::eq(*a.lower_bounds().begin(), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(*a.upper_bounds().begin(), symbolic::integer(std::numeric_limits<uint16_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Int16));
    EXPECT_TRUE(symbolic::eq(*a.lower_bounds().begin(), symbolic::integer(std::numeric_limits<int16_t>::min())));
    EXPECT_TRUE(symbolic::eq(*a.upper_bounds().begin(), symbolic::integer(std::numeric_limits<int16_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::UInt32));
    EXPECT_TRUE(symbolic::eq(*a.lower_bounds().begin(), symbolic::integer(std::numeric_limits<uint32_t>::min())));
    EXPECT_TRUE(symbolic::eq(*a.upper_bounds().begin(), symbolic::integer(std::numeric_limits<uint32_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Int32));
    EXPECT_TRUE(symbolic::eq(*a.lower_bounds().begin(), symbolic::integer(std::numeric_limits<int32_t>::min())));
    EXPECT_TRUE(symbolic::eq(*a.upper_bounds().begin(), symbolic::integer(std::numeric_limits<int32_t>::max())));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::UInt64));
    EXPECT_TRUE(symbolic::eq(*a.lower_bounds().begin(), symbolic::integer(std::numeric_limits<uint64_t>::min())));
    EXPECT_TRUE(symbolic::eq(*a.upper_bounds().begin(), SymEngine::Inf));

    a = symbolic::Assumption::create(x, types::Scalar(types::PrimitiveType::Int64));
    EXPECT_TRUE(symbolic::eq(*a.lower_bounds().begin(), symbolic::integer(std::numeric_limits<int64_t>::min())));
    EXPECT_TRUE(symbolic::eq(*a.upper_bounds().begin(), symbolic::integer(std::numeric_limits<int64_t>::max())));
}

TEST(AssumptionsTest, Constraints_InitiallyEmpty) {
    auto x = symbolic::symbol("x");
    symbolic::Assumption a(x);
    EXPECT_TRUE(a.constraints().empty());
}

TEST(AssumptionsTest, Constraints_AddContainsRemove) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    symbolic::Assumption a(i);

    // c1: i + j - 15  (i.e. i + j <= 15)
    auto c1 = symbolic::sub(symbolic::add(i, j), symbolic::integer(15));
    // c2: 5 - i - j   (i.e. i + j >= 5)
    auto c2 = symbolic::sub(symbolic::integer(5), symbolic::add(i, j));

    a.add_constraint(c1);
    a.add_constraint(c2);
    EXPECT_EQ(a.constraints().size(), 2u);
    EXPECT_TRUE(a.contains_constraint(c1));
    EXPECT_TRUE(a.contains_constraint(c2));

    EXPECT_TRUE(a.remove_constraint(c1));
    EXPECT_EQ(a.constraints().size(), 1u);
    EXPECT_FALSE(a.contains_constraint(c1));
    EXPECT_TRUE(a.contains_constraint(c2));

    // Removing a non-present constraint returns false and leaves the set unchanged.
    EXPECT_FALSE(a.remove_constraint(c1));
    EXPECT_EQ(a.constraints().size(), 1u);
}

TEST(AssumptionsTest, Constraints_DedupByValue) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    symbolic::Assumption a(i);

    // The set is keyed by SymEngine value equality, so `i + j - 15` added
    // twice (constructed independently) should collapse to a single entry.
    auto c = symbolic::sub(symbolic::add(i, j), symbolic::integer(15));
    auto c_dup = symbolic::sub(symbolic::add(i, j), symbolic::integer(15));
    a.add_constraint(c);
    a.add_constraint(c_dup);
    EXPECT_EQ(a.constraints().size(), 1u);
}

TEST(AssumptionsTest, Constraints_CopyAndAssign) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    symbolic::Assumption a(i);
    auto c = symbolic::sub(symbolic::add(i, j), symbolic::integer(15));
    a.add_constraint(c);

    // Copy ctor carries the constraints over.
    symbolic::Assumption b(a);
    EXPECT_EQ(b.constraints().size(), 1u);
    EXPECT_TRUE(b.contains_constraint(c));

    // Mutating the copy does not affect the original (constraints_ is a value-type set).
    auto c2 = symbolic::sub(symbolic::integer(5), symbolic::add(i, j));
    b.add_constraint(c2);
    EXPECT_EQ(a.constraints().size(), 1u);
    EXPECT_EQ(b.constraints().size(), 2u);

    // Assignment operator carries the constraints over and overwrites the LHS set.
    symbolic::Assumption d(i);
    d = b;
    EXPECT_EQ(d.constraints().size(), 2u);
    EXPECT_TRUE(d.contains_constraint(c));
    EXPECT_TRUE(d.contains_constraint(c2));
}
