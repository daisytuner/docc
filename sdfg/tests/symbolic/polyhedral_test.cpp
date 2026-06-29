#include "sdfg/symbolic/polyhedral.h"

#include <gtest/gtest.h>

#include "sdfg/symbolic/assumptions.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace {

// Helper: a loop-style induction variable in [0, ub-1].
symbolic::Assumption make_indvar(const symbolic::Symbol& sym, const symbolic::Symbol& ub) {
    types::Scalar desc(types::PrimitiveType::Int64);
    auto a = symbolic::Assumption::create(sym, desc);
    a.add_lower_bound(symbolic::zero());
    a.add_upper_bound(symbolic::sub(ub, symbolic::one()));
    return a;
}

// Helper: a constant parameter (e.g. a dimension size) >= 1.
symbolic::Assumption make_param(const symbolic::Symbol& sym) {
    types::Scalar desc(types::PrimitiveType::Int64);
    auto a = symbolic::Assumption::create(sym, desc);
    a.add_lower_bound(symbolic::one());
    a.constant(true);
    return a;
}

} // namespace

TEST(PolyhedralEqualOnDomainTest, ScalarTrivial) {
    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    symbolic::Assumptions assums;
    assums.insert({i, make_indvar(i, N)});
    assums.insert({N, make_param(N)});

    // Empty subsets model a scalar access: always the same cell.
    EXPECT_TRUE(symbolic::polyhedral::equal_on_domain({}, {}, i, assums));
}

TEST(PolyhedralEqualOnDomainTest, IdenticalAccess) {
    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    symbolic::Assumptions assums;
    assums.insert({i, make_indvar(i, N)});
    assums.insert({N, make_param(N)});

    EXPECT_TRUE(symbolic::polyhedral::equal_on_domain({i}, {i}, i, assums));
}

TEST(PolyhedralEqualOnDomainTest, ConstantOffsetReassociated) {
    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    symbolic::Assumptions assums;
    assums.insert({i, make_indvar(i, N)});
    assums.insert({N, make_param(N)});

    // i + 1 vs 1 + i
    auto f = symbolic::add(i, symbolic::one());
    auto g = symbolic::add(symbolic::one(), i);
    EXPECT_TRUE(symbolic::polyhedral::equal_on_domain({f}, {g}, i, assums));
}

TEST(PolyhedralEqualOnDomainTest, ParametricLinearizedEqual) {
    auto i = symbolic::symbol("i");
    auto k = symbolic::symbol("k");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    symbolic::Assumptions assums;
    assums.insert({i, make_indvar(i, N)});
    assums.insert({k, make_indvar(k, M)});
    assums.insert({N, make_param(N)});
    assums.insert({M, make_param(M)});

    // i*N + k  ==  k + N*i  (linearized accumulator address, reassociated)
    auto f = symbolic::add(symbolic::mul(i, N), k);
    auto g = symbolic::add(k, symbolic::mul(N, i));
    EXPECT_TRUE(symbolic::polyhedral::equal_on_domain({f}, {g}, i, assums));
}

TEST(PolyhedralEqualOnDomainTest, ShiftedAccessNotEqual) {
    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    symbolic::Assumptions assums;
    assums.insert({i, make_indvar(i, N)});
    assums.insert({N, make_param(N)});

    // A[i] vs A[i-1]: a scan/recurrence, NOT a reduction.
    auto f = i;
    auto g = symbolic::sub(i, symbolic::one());
    EXPECT_FALSE(symbolic::polyhedral::equal_on_domain({f}, {g}, i, assums));
}

TEST(PolyhedralEqualOnDomainTest, DifferentVarsNotEqual) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    symbolic::Assumptions assums;
    assums.insert({i, make_indvar(i, N)});
    assums.insert({j, make_indvar(j, N)});
    assums.insert({N, make_param(N)});

    EXPECT_FALSE(symbolic::polyhedral::equal_on_domain({i}, {j}, i, assums));
}

TEST(PolyhedralEqualOnDomainTest, SizeMismatchNotEqual) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    symbolic::Assumptions assums;
    assums.insert({i, make_indvar(i, N)});
    assums.insert({j, make_indvar(j, N)});
    assums.insert({N, make_param(N)});

    EXPECT_FALSE(symbolic::polyhedral::equal_on_domain({i}, {i, j}, i, assums));
}

TEST(PolyhedralEqualOnDomainTest, MultiDimEqual) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    symbolic::Assumptions assums;
    assums.insert({i, make_indvar(i, N)});
    assums.insert({j, make_indvar(j, M)});
    assums.insert({N, make_param(N)});
    assums.insert({M, make_param(M)});

    EXPECT_TRUE(symbolic::polyhedral::equal_on_domain({i, j}, {i, j}, i, assums));
}

TEST(PolyhedralEqualOnDomainTest, MultiDimPermutedNotEqual) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto N = symbolic::symbol("N");
    auto M = symbolic::symbol("M");
    symbolic::Assumptions assums;
    assums.insert({i, make_indvar(i, N)});
    assums.insert({j, make_indvar(j, M)});
    assums.insert({N, make_param(N)});
    assums.insert({M, make_param(M)});

    // [i, j] vs [j, i] are different cells in general.
    EXPECT_FALSE(symbolic::polyhedral::equal_on_domain({i, j}, {j, i}, i, assums));
}
