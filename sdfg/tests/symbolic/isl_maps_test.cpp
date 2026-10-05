// The direct ISL map construction (`expressions_to_intersection_maps`) must have exactly the semantics of
// parsing the string form (`expressions_to_intersection_map_str`), and both must agree with codegen.

#include <gtest/gtest.h>

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "sdfg/symbolic/maps.h"
#include "sdfg/symbolic/polyhedral.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/symbolic/utils.h"

using namespace sdfg;

namespace {

symbolic::Assumption dim(const std::string& name, int64_t lo, int64_t hi) {
    symbolic::Assumption a(symbolic::symbol(name));
    a.add_lower_bound(symbolic::integer(lo));
    a.add_upper_bound(symbolic::integer(hi));
    return a;
}

symbolic::Assumption param(const std::string& name, int64_t lo) {
    symbolic::Assumption a(symbolic::symbol(name));
    a.constant(true);
    a.add_lower_bound(symbolic::integer(lo));
    return a;
}

symbolic::Assumptions assumptions(std::initializer_list<symbolic::Assumption> list) {
    symbolic::Assumptions assums;
    for (auto& a : list) {
        assums.insert({a.symbol(), a});
    }
    return assums;
}

struct Maps {
    symbolic::polyhedral::IslMap m1, m2, m3;
};

struct Built : Maps {
    bool ok = false;
};

Built direct(
    isl_ctx* ctx,
    const symbolic::MultiExpression& e1,
    const symbolic::MultiExpression& e2,
    const symbolic::Symbol& indvar,
    const symbolic::Assumptions& assums
) {
    isl_map *a, *b, *c;
    Built r;
    r.ok = symbolic::expressions_to_intersection_maps(ctx, e1, e2, indvar, assums, assums, &a, &b, &c);
    r.m1 = symbolic::polyhedral::IslMap(a);
    r.m2 = symbolic::polyhedral::IslMap(b);
    r.m3 = symbolic::polyhedral::IslMap(c);
    return r;
}

Maps parsed(
    isl_ctx* ctx,
    const symbolic::MultiExpression& e1,
    const symbolic::MultiExpression& e2,
    const symbolic::Symbol& indvar,
    const symbolic::Assumptions& assums
) {
    auto s = symbolic::expressions_to_intersection_map_str(e1, e2, indvar, assums, assums);
    Maps r;
    r.m1 = symbolic::polyhedral::IslMap(isl_map_read_from_str(ctx, std::get<0>(s).c_str()));
    r.m2 = symbolic::polyhedral::IslMap(isl_map_read_from_str(ctx, std::get<1>(s).c_str()));
    r.m3 = symbolic::polyhedral::IslMap(isl_map_read_from_str(ctx, std::get<2>(s).c_str()));
    return r;
}

std::vector<std::string> dim_names(isl_map* m, isl_dim_type type) {
    std::vector<std::string> names;
    for (int i = 0; i < isl_map_dim(m, type); i++) {
        const char* n = isl_map_get_dim_name(m, type, i);
        names.push_back(n ? n : "");
    }
    return names;
}

std::string to_str(isl_map* m) {
    char* s = isl_map_to_str(m);
    std::string r = s ? s : "(null)";
    free(s);
    return r;
}

std::string to_str(isl_set* m) {
    char* s = isl_set_to_str(m);
    std::string r = s ? s : "(null)";
    free(s);
    return r;
}

void expect_same_map(isl_map* actual, isl_map* expected, const std::string& what) {
    ASSERT_NE(actual, nullptr) << what;
    ASSERT_NE(expected, nullptr) << what;
    EXPECT_EQ(isl_map_is_equal(actual, expected), isl_bool_true)
        << what << "\n  direct: " << to_str(actual) << "\n  parsed: " << to_str(expected);
    EXPECT_EQ(dim_names(actual, isl_dim_param), dim_names(expected, isl_dim_param)) << what;
    EXPECT_EQ(dim_names(actual, isl_dim_in), dim_names(expected, isl_dim_in)) << what;
}

int64_t c_div(int64_t a, int64_t b) {
    return a / b;
}
int64_t c_mod(int64_t a, int64_t b) {
    return a % b;
}

class IslMapsTest : public ::testing::Test {
protected:
    symbolic::polyhedral::IslCtx ctx;
    isl_ctx* c() {
        return ctx.get();
    }
};

} // namespace

/***** Direct construction == parsing *****/

TEST_F(IslMapsTest, DirectMatchesParsedOnCorpus) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto n = symbolic::symbol("N");
    auto assums = assumptions({dim("i", -9, 9), dim("j", 0, 5), param("N", 1)});
    auto idiv = [](symbolic::Expression a, int64_t b) {
        return SymEngine::function_symbol("idiv", {a, symbolic::integer(b)});
    };
    auto imod = [](symbolic::Expression a, int64_t b) {
        return SymEngine::function_symbol("imod", {a, symbolic::integer(b)});
    };

    std::vector<symbolic::Expression> corpus = {
        i,
        symbolic::add(symbolic::sub(symbolic::mul(symbolic::integer(3), i), symbolic::mul(symbolic::integer(2), j)), n),
        idiv(i, 4),
        idiv(i, -4),
        imod(i, 3),
        imod(i, -3),
        symbolic::abs(symbolic::sub(i, symbolic::integer(2))),
        symbolic::max(i, j),
        symbolic::min(i, n),
        SymEngine::max({symbolic::zero(), symbolic::sub(i, symbolic::one()), j}),
        symbolic::add(idiv(symbolic::max(i, symbolic::zero()), 2), imod(j, 4)),
        symbolic::mul(symbolic::integer(2), idiv(symbolic::add(i, j), 3)),
    };
    for (auto& e : corpus) {
        for (auto& indvar : {i, j}) {
            SCOPED_TRACE(e->__str__() + " indvar " + indvar->get_name());
            auto d = direct(c(), {e, j}, {i, e}, indvar, assums);
            auto p = parsed(c(), {e, j}, {i, e}, indvar, assums);
            ASSERT_TRUE(d.ok);
            expect_same_map(d.m1.get(), p.m1.get(), "map_1");
            expect_same_map(d.m2.get(), p.m2.get(), "map_2");
            expect_same_map(d.m3.get(), p.m3.get(), "map_3");
        }
    }
}

TEST_F(IslMapsTest, SymbolicBoundsMatchParsed) {
    // Bounds referencing other dims and parameters, including the skewed `max` bounds of tiled loops.
    auto t = symbolic::symbol("t");
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("N");
    symbolic::Assumption a_i(i);
    a_i.add_lower_bound(
        symbolic::max(symbolic::zero(), symbolic::sub(symbolic::mul(symbolic::integer(32), t), symbolic::one()))
    );
    a_i.add_upper_bound(
        symbolic::
            min(symbolic::sub(n, symbolic::one()),
                symbolic::add(symbolic::mul(symbolic::integer(32), t), symbolic::integer(33)))
    );
    auto assums = assumptions({dim("t", 0, 7), a_i, param("N", 3)});

    auto d = direct(c(), {i}, {symbolic::add(i, symbolic::one())}, t, assums);
    auto p = parsed(c(), {i}, {symbolic::add(i, symbolic::one())}, t, assums);
    ASSERT_TRUE(d.ok);
    expect_same_map(d.m1.get(), p.m1.get(), "map_1");
    expect_same_map(d.m2.get(), p.m2.get(), "map_2");
    expect_same_map(d.m3.get(), p.m3.get(), "map_3");
}

/***** Semantics agree with C evaluation, point by point *****/

TEST_F(IslMapsTest, AccessValuesMatchC) {
    auto i = symbolic::symbol("i");
    auto assums = assumptions({dim("i", -12, 12)});
    struct Case {
        symbolic::Expression expr;
        std::function<int64_t(int64_t)> eval;
    };
    std::vector<Case> cases = {
        {SymEngine::function_symbol("idiv", {i, symbolic::integer(4)}),
         [](int64_t v) {
             return c_div(v, 4);
         }},
        {SymEngine::function_symbol("idiv", {i, symbolic::integer(-4)}),
         [](int64_t v) {
             return c_div(v, -4);
         }},
        {SymEngine::function_symbol("imod", {i, symbolic::integer(5)}),
         [](int64_t v) {
             return c_mod(v, 5);
         }},
        {SymEngine::function_symbol("imod", {i, symbolic::integer(-5)}),
         [](int64_t v) {
             return c_mod(v, -5);
         }},
        {symbolic::abs(symbolic::add(i, symbolic::integer(3))),
         [](int64_t v) {
             return std::abs(v + 3);
         }},
        {symbolic::max(symbolic::sub(i, symbolic::integer(2)), symbolic::integer(-1)),
         [](int64_t v) {
             return std::max<int64_t>(v - 2, -1);
         }},
        {symbolic::min(symbolic::mul(symbolic::integer(2), i), symbolic::integer(5)),
         [](int64_t v) {
             return std::min<int64_t>(2 * v, 5);
         }},
        {symbolic::
             add(SymEngine::function_symbol("idiv", {symbolic::sub(i, symbolic::integer(7)), symbolic::integer(3)}),
                 SymEngine::function_symbol("imod", {i, symbolic::integer(2)})),
         [](int64_t v) {
             return c_div(v - 7, 3) + c_mod(v, 2);
         }},
    };
    for (auto& tc : cases) {
        SCOPED_TRACE(tc.expr->__str__());
        auto d = direct(c(), {tc.expr}, {tc.expr}, i, assums);
        ASSERT_TRUE(d.ok);
        std::string expected = "{ [i_1] -> [o] : ";
        for (int64_t v = -12; v <= 12; ++v) {
            expected += (v == -12 ? "" : " or ") + std::string("(i_1 = ") + std::to_string(v) +
                        " and o = " + std::to_string(tc.eval(v)) + ")";
        }
        expected += " }";
        symbolic::polyhedral::IslMap reference(isl_map_read_from_str(c(), expected.c_str()));
        ASSERT_TRUE(reference);
        EXPECT_EQ(isl_map_is_equal(d.m1.get(), reference.get()), isl_bool_true) << to_str(d.m1.get());
    }
}

TEST_F(IslMapsTest, StrideConstraint) {
    // i = 1, 3, 5, ... from `i = i + 2` with tight lower bound 1.
    auto i = symbolic::symbol("i");
    auto a_i = dim("i", 1, 9);
    a_i.map(symbolic::add(i, symbolic::integer(2)));
    a_i.tight_lower_bound(symbolic::one());
    auto assums = assumptions({a_i});

    auto d = direct(c(), {i}, {i}, i, assums);
    auto p = parsed(c(), {i}, {i}, i, assums);
    ASSERT_TRUE(d.ok);
    expect_same_map(d.m1.get(), p.m1.get(), "map_1");
    symbolic::polyhedral::IslSet domain(isl_map_domain(isl_map_copy(d.m1.get())));
    symbolic::polyhedral::IslSet
        odd(isl_set_read_from_str(c(), "{ [i_1] : i_1 = 1 or i_1 = 3 or i_1 = 5 or i_1 = 7 or i_1 = 9 }"));
    EXPECT_EQ(isl_set_is_equal(domain.get(), odd.get()), isl_bool_true) << to_str(domain.get());
}

TEST_F(IslMapsTest, MonotonicityMap) {
    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto assums = assumptions({dim("i", 0, 9), dim("j", 0, 9)});

    auto forward = direct(c(), {i, j}, {i, j}, i, assums);
    ASSERT_TRUE(forward.ok);
    symbolic::polyhedral::IslMap expected(isl_map_read_from_str(c(), "{ [i_2, j_2] -> [i_1, j_1] : i_1 < i_2 }"));
    EXPECT_EQ(isl_map_is_equal(forward.m3.get(), expected.get()), isl_bool_true) << to_str(forward.m3.get());

    // An indvar that is not a dimension imposes no order.
    auto unordered = direct(c(), {i, j}, {i, j}, k, assums);
    ASSERT_TRUE(unordered.ok);
    symbolic::polyhedral::IslMap universe(isl_map_read_from_str(c(), "{ [i_2, j_2] -> [i_1, j_1] }"));
    EXPECT_EQ(isl_map_is_equal(unordered.m3.get(), universe.get()), isl_bool_true) << to_str(unordered.m3.get());
}

/***** Unsupported inputs decline, and the string form rejects them as well *****/

TEST_F(IslMapsTest, NonAffineInputsAreDeclined) {
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("N");
    auto assums = assumptions({dim("i", 0, 9), param("N", 1)});
    std::vector<symbolic::Expression> non_affine = {
        symbolic::mul(n, i),
        symbolic::mul(i, i),
        SymEngine::function_symbol("idiv", {i, n}),
        SymEngine::function_symbol("imod", {i, symbolic::zero()}),
        symbolic::zext_i64(i),
        SymEngine::div(i, symbolic::integer(2)),
    };
    for (auto& e : non_affine) {
        SCOPED_TRACE(e->__str__());
        auto d = direct(c(), {e}, {e}, i, assums);
        EXPECT_FALSE(d.ok);
        EXPECT_FALSE(d.m1 || d.m2 || d.m3);
        EXPECT_FALSE(parsed(c(), {e}, {e}, i, assums).m1);
    }
}

TEST_F(IslMapsTest, NameClashIsDeclined) {
    // The suffixed dimension `i_1` would collide with the symbol `i_1`.
    auto i = symbolic::symbol("i");
    auto i_1 = symbolic::symbol("i_1");
    auto assums = assumptions({dim("i", 0, 9), dim("i_1", 0, 9)});
    auto d = direct(c(), {symbolic::add(i, i_1)}, {i}, i, assums);
    EXPECT_FALSE(d.ok);
    EXPECT_FALSE(d.m1 || d.m2 || d.m3);
}

/***** The string printer emits ISL syntax *****/

TEST_F(IslMapsTest, PrinterEmitsIslSyntax) {
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("N");
    auto eq = symbolic::constraint_to_isl_str(symbolic::Eq(i, n));
    EXPECT_TRUE(eq == "i = N" || eq == "N = i") << eq;
    auto le =
        symbolic::constraint_to_isl_str(symbolic::Le(symbolic::max(i, n), symbolic::min(n, symbolic::integer(4))));
    EXPECT_NE(le.find("max("), std::string::npos) << le;
    EXPECT_NE(le.find("min("), std::string::npos) << le;
    EXPECT_EQ(le.find("fmax"), std::string::npos) << le;
    EXPECT_EQ(le.find("fmin"), std::string::npos) << le;

    auto assums = assumptions({dim("i", -3, 3)});
    auto map = symbolic::expression_to_map_str({symbolic::max(i, symbolic::zero())}, assums);
    symbolic::polyhedral::IslMap parsed_map(isl_map_read_from_str(c(), map.c_str()));
    EXPECT_TRUE(parsed_map) << map;
}

/***** End to end: dependence distances under `max` bounds are now computed *****/

TEST_F(IslMapsTest, DependenceDeltasUnderMaxBound) {
    // for i in [max(0, N - 5), N]: A[i] = A[i - 1]  ->  distance 1 (previously unknown: `fmax` did not parse).
    auto i = symbolic::symbol("i");
    auto n = symbolic::symbol("N");
    symbolic::Assumption a_i(i);
    a_i.add_lower_bound(symbolic::max(symbolic::zero(), symbolic::sub(n, symbolic::integer(5))));
    a_i.add_upper_bound(n);
    auto assums = assumptions({a_i, param("N", 0)});

    auto deltas = symbolic::maps::dependence_deltas({i}, {symbolic::sub(i, symbolic::one())}, i, assums, assums);
    ASSERT_FALSE(deltas.empty);
    ASSERT_FALSE(deltas.deltas_str.empty());
    ASSERT_EQ(deltas.dimensions, std::vector<std::string>{"i"});
    symbolic::polyhedral::IslSet set(isl_set_read_from_str(c(), deltas.deltas_str.c_str()));
    symbolic::polyhedral::IslSet one(isl_set_read_from_str(c(), "{ [d] : d = 1 }"));
    ASSERT_TRUE(set);
    EXPECT_EQ(isl_set_is_subset(set.get(), one.get()), isl_bool_true) << deltas.deltas_str;
    EXPECT_EQ(isl_set_is_empty(set.get()), isl_bool_false) << deltas.deltas_str;
}
