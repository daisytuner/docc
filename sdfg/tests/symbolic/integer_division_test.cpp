// idiv/imod are defined as signed C/C++ `/` and `%` on int64 (truncation toward zero, remainder with the
// sign of the dividend). These tests pin that definition across folding, simplification, derived floor/ceil
// divisions, bound analysis, C codegen and the ISL encoding.

#include <gtest/gtest.h>

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "sdfg/builder/sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cpp_language_extension.h"
#include "sdfg/symbolic/extreme_values.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/symbolic/utils.h"

using namespace sdfg;

namespace {

int64_t c_div(int64_t a, int64_t b) {
    return a / b;
}
int64_t c_mod(int64_t a, int64_t b) {
    return a % b;
}

int64_t math_floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

int64_t math_ceil_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && ((a < 0) == (b < 0))) ? q + 1 : q;
}

std::optional<int64_t> literal(const symbolic::Expression& e) {
    if (!SymEngine::is_a<SymEngine::Integer>(*e)) {
        return std::nullopt;
    }
    return SymEngine::down_cast<const SymEngine::Integer&>(*e).as_int();
}

bool is_function(const symbolic::Expression& e, const std::string& name) {
    return SymEngine::is_a<SymEngine::FunctionSymbol>(*e) &&
           SymEngine::down_cast<const SymEngine::FunctionSymbol&>(*e).get_name() == name;
}

// Reference evaluator for closed integer expressions with codegen (C) semantics.
int64_t eval(const symbolic::Expression& e) {
    if (auto v = literal(e)) {
        return *v;
    }
    const auto& args = e->get_args();
    if (SymEngine::is_a<SymEngine::Add>(*e)) {
        int64_t r = 0;
        for (auto& a : args) {
            r += eval(a);
        }
        return r;
    }
    if (SymEngine::is_a<SymEngine::Mul>(*e)) {
        int64_t r = 1;
        for (auto& a : args) {
            r *= eval(a);
        }
        return r;
    }
    if (SymEngine::is_a<SymEngine::Max>(*e) || SymEngine::is_a<SymEngine::Min>(*e)) {
        bool is_max = SymEngine::is_a<SymEngine::Max>(*e);
        int64_t r = eval(args[0]);
        for (auto& a : args) {
            r = is_max ? std::max(r, eval(a)) : std::min(r, eval(a));
        }
        return r;
    }
    if (is_function(e, "idiv")) {
        return c_div(eval(args[0]), eval(args[1]));
    }
    if (is_function(e, "imod")) {
        return c_mod(eval(args[0]), eval(args[1]));
    }
    ADD_FAILURE() << "cannot evaluate " << e->__str__();
    return 0;
}

int64_t eval_at(const symbolic::Expression& e, const symbolic::ExpressionMapping& values) {
    return eval(SymEngine::subs(e, values));
}

const std::vector<int64_t> kDividends = {-17, -16, -9, -8, -7, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17};
const std::vector<int64_t> kDivisors = {-8, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 8};
const std::vector<int64_t> kPositiveDivisors = {1, 2, 3, 4, 5, 8};

} // namespace

/***** Literal folding matches C exactly *****/

TEST(IntegerDivisionTest, LiteralFoldingMatchesC) {
    for (int64_t a : kDividends) {
        for (int64_t b : kDivisors) {
            SCOPED_TRACE(std::to_string(a) + " / " + std::to_string(b));
            auto ia = symbolic::integer(a);
            auto ib = symbolic::integer(b);
            EXPECT_EQ(literal(symbolic::div(ia, ib)), c_div(a, b));
            EXPECT_EQ(literal(symbolic::mod(ia, ib)), c_mod(a, b));
            EXPECT_EQ(literal(symbolic::floor_div(ia, ib)), math_floor_div(a, b));
            EXPECT_EQ(literal(symbolic::ceil_div(ia, ib)), math_ceil_div(a, b));
            if (b > 0) {
                EXPECT_EQ(literal(symbolic::ceil_count(ia, ib)), std::max<int64_t>(0, math_ceil_div(a, b)));
            }
        }
    }
}

TEST(IntegerDivisionTest, DivisionIdentity) {
    for (int64_t a : kDividends) {
        for (int64_t b : kDivisors) {
            auto q = *literal(symbolic::div(symbolic::integer(a), symbolic::integer(b)));
            auto r = *literal(symbolic::mod(symbolic::integer(a), symbolic::integer(b)));
            EXPECT_EQ(b * q + r, a) << a << ", " << b;
            EXPECT_TRUE(r == 0 || (r < 0) == (a < 0)) << "remainder must take the dividend's sign";
            EXPECT_LT(std::abs(r), std::abs(b));
        }
    }
}

TEST(IntegerDivisionTest, ExtremeValuesFold) {
    const int64_t min = std::numeric_limits<int64_t>::min();
    const int64_t max = std::numeric_limits<int64_t>::max();
    EXPECT_EQ(literal(symbolic::div(symbolic::integer(max), symbolic::integer(-1))), -max);
    EXPECT_EQ(literal(symbolic::div(symbolic::integer(min), symbolic::integer(2))), min / 2);
    EXPECT_EQ(literal(symbolic::mod(symbolic::integer(min), symbolic::integer(3))), min % 3);
    EXPECT_EQ(literal(symbolic::div(symbolic::integer(min), symbolic::integer(max))), min / max);
}

/***** Undefined behavior in C is never folded *****/

TEST(IntegerDivisionTest, DivisionByZeroIsNotFolded) {
    auto x = symbolic::symbol("x");
    EXPECT_TRUE(is_function(symbolic::div(symbolic::integer(5), symbolic::zero()), "idiv"));
    EXPECT_TRUE(is_function(symbolic::div(symbolic::zero(), symbolic::zero()), "idiv"));
    EXPECT_TRUE(is_function(symbolic::div(x, symbolic::zero()), "idiv"));
    EXPECT_TRUE(is_function(symbolic::mod(symbolic::integer(5), symbolic::zero()), "imod"));
    EXPECT_TRUE(is_function(symbolic::mod(symbolic::zero(), symbolic::zero()), "imod"));
    EXPECT_TRUE(is_function(
        symbolic::simplify(SymEngine::function_symbol("imod", {symbolic::integer(5), symbolic::zero()})), "imod"
    ));
}

TEST(IntegerDivisionTest, MinOverMinusOneIsNotFolded) {
    auto min = symbolic::integer(std::numeric_limits<int64_t>::min());
    auto minus_one = symbolic::integer(-1);
    EXPECT_TRUE(is_function(symbolic::div(min, minus_one), "idiv"));
    EXPECT_TRUE(is_function(symbolic::mod(min, minus_one), "imod"));
    EXPECT_TRUE(is_function(symbolic::simplify(SymEngine::function_symbol("idiv", {min, minus_one})), "idiv"));
}

TEST(IntegerDivisionTest, OutOfRangeLiteralsAreNotFolded) {
    auto huge = SymEngine::integer(SymEngine::integer_class("100000000000000000000"));
    EXPECT_TRUE(is_function(symbolic::div(huge, symbolic::integer(3)), "idiv"));
    EXPECT_TRUE(is_function(symbolic::mod(huge, symbolic::integer(3)), "imod"));
}

/***** Symbolic identities *****/

TEST(IntegerDivisionTest, SymbolicIdentities) {
    auto x = symbolic::symbol("x");
    auto minus_x = symbolic::mul(symbolic::integer(-1), x);
    EXPECT_TRUE(symbolic::eq(symbolic::div(x, symbolic::one()), x));
    EXPECT_TRUE(symbolic::eq(symbolic::div(x, symbolic::integer(-1)), minus_x));
    EXPECT_TRUE(symbolic::eq(symbolic::div(symbolic::zero(), x), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(symbolic::div(x, x), symbolic::one()));
    EXPECT_TRUE(symbolic::eq(symbolic::mod(x, symbolic::one()), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(symbolic::mod(x, symbolic::integer(-1)), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(symbolic::mod(symbolic::zero(), x), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(symbolic::mod(x, x), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(symbolic::mod(minus_x, x), symbolic::zero()));

    // No floor-semantics rewrites: idiv(x, 2) is not x/2 rounded down for negative x.
    EXPECT_TRUE(is_function(symbolic::div(x, symbolic::integer(2)), "idiv"));
    EXPECT_TRUE(is_function(symbolic::mod(x, symbolic::integer(2)), "imod"));
}

TEST(IntegerDivisionTest, SymbolicIdentitiesHoldAtEveryValue) {
    auto x = symbolic::symbol("x");
    std::vector<symbolic::Expression> exprs = {
        symbolic::div(x, symbolic::integer(-1)),
        symbolic::mod(x, symbolic::integer(-1)),
        symbolic::mod(symbolic::mul(symbolic::integer(-1), x), x),
    };
    for (int64_t v : kDividends) {
        if (v == 0) {
            continue;
        }
        symbolic::ExpressionMapping at{{x, symbolic::integer(v)}};
        EXPECT_EQ(eval_at(exprs[0], at), c_div(v, -1));
        EXPECT_EQ(eval_at(exprs[1], at), c_mod(v, -1));
        EXPECT_EQ(eval_at(exprs[2], at), c_mod(-v, v));
    }
}

/***** simplify *****/

TEST(IntegerDivisionTest, SimplifyFoldsLikeC) {
    for (int64_t a : kDividends) {
        for (int64_t b : kDivisors) {
            auto d =
                symbolic::simplify(SymEngine::function_symbol("idiv", {symbolic::integer(a), symbolic::integer(b)}));
            auto m =
                symbolic::simplify(SymEngine::function_symbol("imod", {symbolic::integer(a), symbolic::integer(b)}));
            EXPECT_EQ(literal(d), c_div(a, b)) << a << " / " << b;
            EXPECT_EQ(literal(m), c_mod(a, b)) << a << " % " << b;
        }
    }
}

TEST(IntegerDivisionTest, SimplifySmallNonNegativeDividend) {
    auto idiv = [](int64_t a, int64_t b) {
        return SymEngine::function_symbol("idiv", {symbolic::integer(a), symbolic::integer(b)});
    };
    // Regression: `lhs < rhs` alone folded idiv(-5, 4) to 0, but C gives -1.
    EXPECT_EQ(literal(symbolic::simplify(idiv(-5, 4))), -1);
    EXPECT_EQ(literal(symbolic::simplify(idiv(3, 4))), 0);

    auto x = symbolic::symbol("x");
    auto two_plus_x_squared = symbolic::add(symbolic::integer(2), symbolic::mul(x, x));
    // Not provably in [0, rhs): stays symbolic.
    EXPECT_TRUE(is_function(symbolic::simplify(SymEngine::function_symbol("idiv", {x, symbolic::integer(4)})), "idiv"));
    EXPECT_TRUE(is_function(
        symbolic::simplify(SymEngine::function_symbol("imod", {two_plus_x_squared, symbolic::integer(4)})), "imod"
    ));
}

TEST(IntegerDivisionTest, SimplifyExactMultiples) {
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto idiv = [](symbolic::Expression a, int64_t b) {
        return symbolic::simplify(SymEngine::function_symbol("idiv", {a, symbolic::integer(b)}));
    };
    auto imod = [](symbolic::Expression a, int64_t b) {
        return symbolic::simplify(SymEngine::function_symbol("imod", {a, symbolic::integer(b)}));
    };
    EXPECT_TRUE(symbolic::eq(idiv(symbolic::mul(symbolic::integer(4), x), 4), x));
    EXPECT_TRUE(symbolic::eq(idiv(symbolic::mul(symbolic::integer(8), x), 4), symbolic::mul(symbolic::integer(2), x)));
    EXPECT_TRUE(symbolic::eq(idiv(symbolic::mul(symbolic::integer(-8), x), 4), symbolic::mul(symbolic::integer(-2), x)));
    EXPECT_TRUE(symbolic::eq(idiv(symbolic::mul(symbolic::integer(8), x), -4), symbolic::mul(symbolic::integer(-2), x)));
    EXPECT_TRUE(
        symbolic::
            eq(idiv(symbolic::mul(symbolic::integer(6), symbolic::mul(x, y)), 3),
               symbolic::mul(symbolic::integer(2), symbolic::mul(x, y)))
    );
    EXPECT_TRUE(symbolic::eq(imod(symbolic::mul(symbolic::integer(6), x), 3), symbolic::zero()));
    EXPECT_TRUE(symbolic::eq(imod(symbolic::mul(symbolic::integer(-6), x), 3), symbolic::zero()));
    // Not a multiple: must stay.
    EXPECT_TRUE(is_function(idiv(symbolic::mul(symbolic::integer(6), x), 4), "idiv"));
    EXPECT_TRUE(is_function(imod(symbolic::mul(symbolic::integer(6), x), 4), "imod"));

    for (int64_t v : kDividends) {
        symbolic::ExpressionMapping at{{x, symbolic::integer(v)}};
        EXPECT_EQ(eval_at(idiv(symbolic::mul(symbolic::integer(-8), x), 4), at), c_div(-8 * v, 4));
        EXPECT_EQ(eval_at(idiv(symbolic::mul(symbolic::integer(8), x), -4), at), c_div(8 * v, -4));
    }
}

/***** Derived floor/ceil divisions over symbolic operands *****/

TEST(IntegerDivisionTest, FloorDivSymbolicDividend) {
    auto x = symbolic::symbol("x");
    for (int64_t b : kDivisors) {
        auto expr = symbolic::floor_div(x, symbolic::integer(b));
        for (int64_t v : kDividends) {
            EXPECT_EQ(eval_at(expr, {{x, symbolic::integer(v)}}), math_floor_div(v, b)) << v << " floor/ " << b;
        }
    }
}

TEST(IntegerDivisionTest, FloorDivSymbolicDivisor) {
    auto x = symbolic::symbol("x");
    auto n = symbolic::symbol("n");
    auto expr = symbolic::floor_div(x, n);
    for (int64_t b : kPositiveDivisors) {
        for (int64_t v : kDividends) {
            EXPECT_EQ(eval_at(expr, {{x, symbolic::integer(v)}, {n, symbolic::integer(b)}}), math_floor_div(v, b))
                << v << " floor/ " << b;
        }
    }
}

TEST(IntegerDivisionTest, CeilDivSymbolicDividend) {
    auto x = symbolic::symbol("x");
    for (int64_t b : kDivisors) {
        auto expr = symbolic::ceil_div(x, symbolic::integer(b));
        for (int64_t v : kDividends) {
            EXPECT_EQ(eval_at(expr, {{x, symbolic::integer(v)}}), math_ceil_div(v, b)) << v << " ceil/ " << b;
        }
    }
}

TEST(IntegerDivisionTest, CeilDivSymbolicDivisor) {
    auto x = symbolic::symbol("x");
    auto n = symbolic::symbol("n");
    auto expr = symbolic::ceil_div(x, n);
    auto clamped = symbolic::ceil_count(x, n);
    for (int64_t b : kPositiveDivisors) {
        for (int64_t v : kDividends) {
            symbolic::ExpressionMapping at{{x, symbolic::integer(v)}, {n, symbolic::integer(b)}};
            EXPECT_EQ(eval_at(expr, at), math_ceil_div(v, b)) << v << " ceil/ " << b;
            EXPECT_EQ(eval_at(clamped, at), std::max<int64_t>(0, math_ceil_div(v, b))) << v << " ceil/ " << b;
        }
    }
}

TEST(IntegerDivisionTest, CeilDivConstantDividendSymbolicDivisor) {
    // Regression: a literal dividend used to produce SymEngine's real-valued ceiling(10/n).
    auto n = symbolic::symbol("n");
    auto expr = symbolic::ceil_div(symbolic::integer(10), n);
    for (int64_t b : kPositiveDivisors) {
        EXPECT_EQ(eval_at(expr, {{n, symbolic::integer(b)}}), math_ceil_div(10, b));
    }
}

TEST(IntegerDivisionTest, CeilCountKeepsCountShape) {
    auto x = symbolic::symbol("x");
    auto expected =
        symbolic::max(symbolic::zero(), symbolic::div(symbolic::add(x, symbolic::integer(31)), symbolic::integer(32)));
    EXPECT_TRUE(symbolic::eq(symbolic::ceil_count(x, symbolic::integer(32)), expected));
}

/***** Bound analysis is sound for both signs *****/

TEST(IntegerDivisionTest, BoundsEncloseEveryValue) {
    auto a = symbolic::symbol("a");
    const std::vector<std::pair<int64_t, int64_t>> ranges = {
        {-17, -9}, {-9, -1}, {-5, 5}, {-3, 2}, {0, 3}, {1, 17}, {5, 7}, {-7, -5}, {6, 9}, {-2, -2}
    };
    for (auto [lo, hi] : ranges) {
        symbolic::Assumption assum(a);
        assum.add_lower_bound(symbolic::integer(lo));
        assum.add_upper_bound(symbolic::integer(hi));
        symbolic::Assumptions assums{{a, assum}};
        for (int64_t b : kDivisors) {
            for (bool is_div : {true, false}) {
                auto expr = is_div ? symbolic::div(a, symbolic::integer(b)) : symbolic::mod(a, symbolic::integer(b));
                auto lb = literal(symbolic::minimum(expr, {}, assums, false));
                auto ub = literal(symbolic::maximum(expr, {}, assums, false));
                ASSERT_TRUE(lb && ub) << expr->__str__() << " on [" << lo << ", " << hi << "]";
                int64_t actual_min = std::numeric_limits<int64_t>::max();
                int64_t actual_max = std::numeric_limits<int64_t>::min();
                for (int64_t v = lo; v <= hi; ++v) {
                    int64_t r = is_div ? c_div(v, b) : c_mod(v, b);
                    actual_min = std::min(actual_min, r);
                    actual_max = std::max(actual_max, r);
                }
                EXPECT_LE(*lb, actual_min) << expr->__str__() << " on [" << lo << ", " << hi << "]";
                EXPECT_GE(*ub, actual_max) << expr->__str__() << " on [" << lo << ", " << hi << "]";
                if (is_div) {
                    // Truncation is monotone, so endpoint bounds are exact.
                    EXPECT_EQ(*lb, actual_min);
                    EXPECT_EQ(*ub, actual_max);
                }
            }
        }
    }
}

TEST(IntegerDivisionTest, ImodBoundsWithUnknownSign) {
    auto a = symbolic::symbol("a");
    symbolic::Assumptions assums{{a, symbolic::Assumption(a)}};
    auto expr = symbolic::mod(a, symbolic::integer(-8));
    EXPECT_EQ(literal(symbolic::minimum(expr, {}, assums, false)), -7);
    EXPECT_EQ(literal(symbolic::maximum(expr, {}, assums, false)), 7);
}

TEST(IntegerDivisionTest, ImodSymbolicRangeDoesNotAssumeNoWrap) {
    // Regression: a symbolic range narrower than the modulus may still wrap (N % 4 == 3), so the bounds
    // must not be imod(N, 4)..imod(N + 2, 4).
    auto a = symbolic::symbol("a");
    auto n = symbolic::symbol("N");
    symbolic::Assumption assum(a);
    assum.add_lower_bound(n);
    assum.add_upper_bound(symbolic::add(n, symbolic::integer(2)));
    symbolic::Assumption assum_n(n);
    assum_n.add_lower_bound(symbolic::zero());
    symbolic::Assumptions assums{{a, assum}, {n, assum_n}};
    auto expr = symbolic::mod(a, symbolic::integer(4));
    EXPECT_EQ(literal(symbolic::minimum(expr, {}, assums, false)), 0);
    EXPECT_EQ(literal(symbolic::maximum(expr, {}, assums, false)), 3);
}

/***** Codegen emits C operators *****/

TEST(IntegerDivisionTest, CodegenEmitsCOperators) {
    builder::SDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    codegen::CLanguageExtension c(sdfg);
    codegen::CPPLanguageExtension cpp(sdfg);
    auto x = symbolic::symbol("x");
    auto q = SymEngine::function_symbol("idiv", {x, symbolic::integer(4)});
    auto r = SymEngine::function_symbol("imod", {x, symbolic::integer(4)});
    EXPECT_EQ(c.expression(q), "((x) / (4))");
    EXPECT_EQ(c.expression(r), "((x) % (4))");
    EXPECT_EQ(cpp.expression(q), "((x) / (4))");
    EXPECT_EQ(cpp.expression(r), "((x) % (4))");
}

/***** The ISL encoding agrees with C *****/

TEST(IntegerDivisionTest, IslEncodingMatchesC) {
    auto x = symbolic::symbol("x");
    symbolic::Assumption assum(x);
    assum.constant(false);
    assum.add_lower_bound(symbolic::integer(-17));
    assum.add_upper_bound(symbolic::integer(17));
    symbolic::Assumptions assums{{x, assum}};

    isl_ctx* ctx = isl_ctx_alloc();
    isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
    for (int64_t b : kDivisors) {
        for (bool is_div : {true, false}) {
            auto expr = SymEngine::function_symbol(is_div ? "idiv" : "imod", {x, symbolic::integer(b)});
            std::string expected = "{ [x] -> [o] : ";
            for (int64_t v = -17; v <= 17; ++v) {
                expected += (v == -17 ? "" : " or ");
                expected += "(x = " + std::to_string(v) +
                            " and o = " + std::to_string(is_div ? c_div(v, b) : c_mod(v, b)) + ")";
            }
            expected += " }";

            auto map_str = symbolic::expression_to_map_str({expr}, assums);
            isl_map* actual = isl_map_read_from_str(ctx, map_str.c_str());
            isl_map* reference = isl_map_read_from_str(ctx, expected.c_str());
            ASSERT_NE(actual, nullptr) << map_str;
            ASSERT_NE(reference, nullptr);
            EXPECT_EQ(isl_map_is_equal(actual, reference), isl_bool_true) << expr->__str__() << ": " << map_str;
            isl_map_free(actual);
            isl_map_free(reference);
        }
    }
    isl_ctx_free(ctx);
}
