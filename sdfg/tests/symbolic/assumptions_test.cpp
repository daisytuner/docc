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
