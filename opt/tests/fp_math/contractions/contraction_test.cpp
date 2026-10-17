#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "compile_and_load.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/fp_math/passes/contraction_precision.h"
#include "sdfg/fp_math/transformations/ozaki_contraction.h"
#include "sdfg/fp_math/transformations/quantize_contraction.h"
#include "sdfg/transformations/transformation_registry.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;
using math::tensor::MatMulNode;
using types::PrimitiveType;

namespace {

/// Y[m, n] = A[m, k] @ B[k, n] over the argument pointers a, b, y of element type @p type.
MatMulNode& add_matmul(
    builder::StructuredSDFGBuilder& builder,
    PrimitiveType type,
    const symbolic::Expression& m,
    const symbolic::Expression& n,
    const symbolic::Expression& k
) {
    auto& sdfg = builder.subject();
    types::Pointer pointer{types::Scalar(type)};
    builder.add_container("a", pointer, true);
    builder.add_container("b", pointer, true);
    builder.add_container("y", pointer, true);
    auto& block = builder.add_block(sdfg.root());
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& y = builder.add_access(block, "y");
    auto& matmul = static_cast<MatMulNode&>(builder.add_library_node<MatMulNode>(
        block, DebugInfo(), math::tensor::TensorLayout({m, k}), math::tensor::TensorLayout({k, n})
    ));
    builder.add_computational_memlet(block, a, matmul, "A", {}, pointer);
    builder.add_computational_memlet(block, b, matmul, "B", {}, pointer);
    builder.add_computational_memlet(block, y, matmul, "Y", {}, pointer);
    return matmul;
}

std::vector<MatMulNode*> matmuls(structured_control_flow::ControlFlowNode& node) {
    std::vector<MatMulNode*> found;
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        for (auto* library_node : block->dataflow().library_nodes()) {
            if (auto* matmul = dynamic_cast<MatMulNode*>(library_node)) {
                found.push_back(matmul);
            }
        }
    } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); i++) {
            auto inner = matmuls(sequence->at(i));
            found.insert(found.end(), inner.begin(), inner.end());
        }
    } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        return matmuls(loop->root());
    }
    return found;
}

constexpr int64_t M = 6, N = 9, K = 33;

/// Operands with rows and columns of different magnitudes, negative values and an all-zero row of A.
template<typename T>
void fill(std::vector<T>& a, std::vector<T>& b) {
    // Full 53-bit significands, so that dropping bits shows.
    std::mt19937_64 random(12345);
    auto next = [&]() {
        return std::ldexp(static_cast<double>(random() >> 11), -52) - 1.0;
    };
    a.resize(M * K);
    b.resize(K * N);
    for (int64_t i = 0; i < M; i++) {
        for (int64_t k = 0; k < K; k++) {
            a[i * K + k] = i == 1 ? T(0) : static_cast<T>(std::ldexp(next(), static_cast<int>(i) * 7 - 12));
        }
    }
    for (int64_t k = 0; k < K; k++) {
        for (int64_t j = 0; j < N; j++) {
            b[k * N + j] = static_cast<T>(std::ldexp(next(), static_cast<int>(j % 4) * 5 - 3));
        }
    }
}

/// Largest error of @p transform's result, relative to sum_k |A[i, k] B[k, j]|.
template<typename T>
double relative_error(
    const std::string& name,
    const std::function<void(data_flow::LibraryNode&, builder::StructuredSDFGBuilder&)>& transform
) {
    auto type = std::is_same_v<T, float> ? PrimitiveType::Float : PrimitiveType::Double;
    builder::StructuredSDFGBuilder builder(name, FunctionType_CPU);
    auto& matmul = add_matmul(builder, type, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    transform(matmul, builder);
    auto function = reinterpret_cast<void (*)(T*, T*, T*)>(test::compile_and_load(builder));

    std::vector<T> a, b, y(M * N, T(-7));
    fill(a, b);
    function(a.data(), b.data(), y.data());

    double worst = 0;
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            long double exact = 0, magnitude = 0;
            for (int64_t k = 0; k < K; k++) {
                exact += static_cast<long double>(a[i * K + k]) * b[k * N + j];
                magnitude += std::fabs(static_cast<long double>(a[i * K + k]) * b[k * N + j]);
            }
            double error = std::fabs(static_cast<double>(y[i * N + j] - exact));
            if (magnitude == 0) {
                EXPECT_EQ(y[i * N + j], T(0)) << "row " << i << ", column " << j;
            } else {
                worst = std::max(worst, error / static_cast<double>(magnitude));
            }
        }
    }
    std::ostringstream text;
    text << std::scientific << worst;
    ::testing::Test::RecordProperty("relative_error_" + name, text.str());
    return worst;
}

template<typename Transformation, typename... Args>
std::function<void(data_flow::LibraryNode&, builder::StructuredSDFGBuilder&)> applying(Args... args) {
    return [=](data_flow::LibraryNode& node, builder::StructuredSDFGBuilder& builder) {
        analysis::AnalysisManager analysis_manager(builder.subject());
        Transformation transformation(node, args...);
        ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
        transformation.apply(builder, analysis_manager);
        EXPECT_NO_THROW(builder.subject().validate());
    };
}

} // namespace

TEST(QuantizeContractionTest, BuildsInt8MatMul) {
    builder::StructuredSDFGBuilder builder("quantize_structure", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Float, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    applying<fp_math::QuantizeContraction>()(matmul, builder);

    auto found = matmuls(builder.subject().root());
    ASSERT_EQ(found.size(), 1);
    EXPECT_EQ(
        found.front()->types(),
        (math::tensor::ContractionTypes{PrimitiveType::Int8, PrimitiveType::Int32, PrimitiveType::Int32})
    );
}

TEST(QuantizeContractionTest, Rejects) {
    auto rejected = [](PrimitiveType type, const symbolic::Expression& k, int bits) {
        builder::StructuredSDFGBuilder builder("quantize_rejects", FunctionType_CPU);
        auto& matmul = add_matmul(builder, type, symbolic::integer(M), symbolic::integer(N), k);
        analysis::AnalysisManager analysis_manager(builder.subject());
        return !fp_math::QuantizeContraction(matmul, bits).can_be_applied(builder, analysis_manager);
    };
    EXPECT_TRUE(rejected(PrimitiveType::Int32, symbolic::integer(K), 8));
    EXPECT_TRUE(rejected(PrimitiveType::Float, symbolic::symbol("K"), 8));
    EXPECT_TRUE(rejected(PrimitiveType::Float, symbolic::integer(200000), 8));
    EXPECT_TRUE(rejected(PrimitiveType::Float, symbolic::integer(K), 9));
    EXPECT_FALSE(rejected(PrimitiveType::Float, symbolic::integer(K), 4));
}

TEST(QuantizeContractionTest, Float_WithinQuantizationError) {
    // Each product errs by at most ~1/127 of its row's and column's largest magnitudes; measured 3.6e-3.
    EXPECT_LT(relative_error<float>("quantize_float", applying<fp_math::QuantizeContraction>()), 1e-2);
}

TEST(QuantizeContractionTest, JsonRoundTrip) {
    builder::StructuredSDFGBuilder builder("quantize_json", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Float, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    nlohmann::json description;
    fp_math::QuantizeContraction(matmul, 6).to_json(description);
    EXPECT_EQ(description["parameters"]["bits"], 6);
    std::string error;
    EXPECT_TRUE(transformations::verify_transformation_round_trip(builder, description, error)) << error;
}

TEST(OzakiContractionTest, DefaultSplits) {
    builder::StructuredSDFGBuilder builder("ozaki_splits", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Double, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    fp_math::OzakiContraction slicing(matmul, fp_math::OzakiVariant::Slicing);
    EXPECT_EQ(slicing.effective_splits(), 8);
    EXPECT_EQ(slicing.matmuls(), 36);
    fp_math::OzakiContraction modular(matmul, fp_math::OzakiVariant::Modular);
    EXPECT_EQ(modular.effective_splits(), 16);
    EXPECT_EQ(modular.matmuls(), 16);
    EXPECT_GE(fp_math::OzakiContraction::modular_bits(16, K), 55);
}

TEST(OzakiContractionTest, BuildsOneInt8MatMulPerModulus) {
    builder::StructuredSDFGBuilder builder("ozaki_structure", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Float, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Modular, 5)(matmul, builder);

    auto found = matmuls(builder.subject().root());
    ASSERT_EQ(found.size(), 5);
    for (auto* inner : found) {
        EXPECT_EQ(inner->input_type(), PrimitiveType::Int8);
        EXPECT_EQ(inner->output_type(), PrimitiveType::Int32);
    }
}

TEST(OzakiContractionTest, Rejects) {
    builder::StructuredSDFGBuilder builder("ozaki_rejects", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Double, symbolic::integer(M), symbolic::integer(N), symbolic::symbol("K"));
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(fp_math::OzakiContraction(matmul).can_be_applied(builder, analysis_manager));
    EXPECT_FALSE(
        fp_math::OzakiContraction(matmul, fp_math::OzakiVariant::Slicing).can_be_applied(builder, analysis_manager)
    );
}

TEST(OzakiContractionTest, NativeBaseline) {
    auto native = [](data_flow::LibraryNode&, builder::StructuredSDFGBuilder&) {
    };
    // The Ozaki variants below match these: 1.7e-16 and 5.6e-8 measured.
    EXPECT_LT(relative_error<double>("native_double", native), 5e-16);
    EXPECT_LT(relative_error<float>("native_float", native), 1e-7);
}

TEST(OzakiContractionTest, Slicing_Double_AtDoublePrecision) {
    EXPECT_LT(
        relative_error<
            double>("ozaki_slicing_double", applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Slicing, 0)),
        5e-16
    );
}

TEST(OzakiContractionTest, Slicing_Float_AtFloatPrecision) {
    EXPECT_LT(
        relative_error<
            float>("ozaki_slicing_float", applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Slicing, 0)),
        1e-7
    );
}

TEST(OzakiContractionTest, Slicing_FewerSlicesLessAccurate) {
    auto coarse = relative_error<
        double>("ozaki_slicing_2", applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Slicing, 2));
    EXPECT_GT(coarse, 1e-6);
    EXPECT_LT(coarse, 1e-3);
}

TEST(OzakiContractionTest, Modular_Double_AtDoublePrecision) {
    EXPECT_LT(
        relative_error<
            double>("ozaki_modular_double", applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Modular, 0)),
        5e-16
    );
}

TEST(OzakiContractionTest, Modular_Float_AtFloatPrecision) {
    EXPECT_LT(
        relative_error<
            float>("ozaki_modular_float", applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Modular, 0)),
        1e-7
    );
}

TEST(OzakiContractionTest, Modular_FewerModuliLessAccurate) {
    auto coarse = relative_error<
        double>("ozaki_modular_6", applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Modular, 6));
    EXPECT_GT(coarse, 1e-12);
    EXPECT_LT(coarse, 1e-3);
}

TEST(OzakiContractionTest, JsonRoundTrip) {
    builder::StructuredSDFGBuilder builder("ozaki_json", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Double, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    for (auto variant : {fp_math::OzakiVariant::Slicing, fp_math::OzakiVariant::Modular}) {
        nlohmann::json description;
        fp_math::OzakiContraction(matmul, variant, 3).to_json(description);
        std::string error;
        EXPECT_TRUE(transformations::verify_transformation_round_trip(builder, description, error)) << error;
    }
}

namespace {

/// C = alpha A @ B^T + beta C over the argument pointers a[M, K], b[N, K], c[M, N] of Double, as a GEMM node.
math::blas::GEMMNode& add_gemm(builder::StructuredSDFGBuilder& builder, const std::string& alpha, const std::string& beta) {
    auto& sdfg = builder.subject();
    types::Scalar element(PrimitiveType::Double);
    types::Pointer pointer(element);
    builder.add_container("a", pointer, true);
    builder.add_container("b", pointer, true);
    builder.add_container("c", pointer, true);
    auto& block = builder.add_block(sdfg.root());
    auto& gemm = static_cast<math::blas::GEMMNode&>(builder.add_library_node<math::blas::GEMMNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        math::blas::BLAS_Precision::d,
        math::blas::BLAS_Layout::RowMajor,
        math::blas::BLAS_Transpose::No,
        math::blas::BLAS_Transpose::Trans,
        symbolic::integer(M),
        symbolic::integer(N),
        symbolic::integer(K),
        symbolic::integer(K),
        symbolic::integer(K),
        symbolic::integer(N)
    ));
    builder.add_computational_memlet(block, builder.add_access(block, "a"), gemm, "__A", {}, pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "b"), gemm, "__B", {}, pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "c"), gemm, "__C", {}, pointer);
    builder.add_computational_memlet(block, builder.add_constant(block, alpha, element), gemm, "__alpha", {}, element);
    builder.add_computational_memlet(block, builder.add_constant(block, beta, element), gemm, "__beta", {}, element);
    return gemm;
}

/// Largest error of the transformed GEMM, relative to |alpha| sum_k |A B| + |beta C|.
double gemm_relative_error(
    const std::string& name,
    double alpha,
    double beta,
    const std::function<void(data_flow::LibraryNode&, builder::StructuredSDFGBuilder&)>& transform
) {
    builder::StructuredSDFGBuilder builder(name, FunctionType_CPU);
    auto& gemm = add_gemm(builder, std::to_string(alpha), std::to_string(beta));
    transform(gemm, builder);
    auto function = reinterpret_cast<void (*)(double*, double*, double*)>(test::compile_and_load(builder));

    std::vector<double> a, b_row_major;
    fill(a, b_row_major);
    std::vector<double> b(N * K), c(M * N), initial(M * N);
    for (int64_t k = 0; k < K; k++) {
        for (int64_t j = 0; j < N; j++) {
            b[j * K + k] = b_row_major[k * N + j];
        }
    }
    for (int64_t e = 0; e < M * N; e++) {
        // BLAS ignores C if beta is 0, even NaNs.
        initial[e] = c[e] = beta == 0 ? std::nan("")
                                      : std::ldexp(static_cast<double>(e % 7) - 3.0, static_cast<int>(e % 5));
    }
    function(a.data(), b.data(), c.data());

    double worst = 0;
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            long double exact = 0, magnitude = 0;
            for (int64_t k = 0; k < K; k++) {
                exact += static_cast<long double>(a[i * K + k]) * b[j * K + k];
                magnitude += std::fabs(static_cast<long double>(a[i * K + k]) * b[j * K + k]);
            }
            exact *= alpha;
            magnitude *= std::fabs(alpha);
            if (beta != 0) {
                exact += beta * static_cast<long double>(initial[i * N + j]);
                magnitude += std::fabs(beta * static_cast<long double>(initial[i * N + j]));
            }
            EXPECT_FALSE(std::isnan(c[i * N + j])) << "row " << i << ", column " << j;
            if (magnitude > 0) {
                worst = std::max(worst, static_cast<double>(std::fabs(c[i * N + j] - exact) / magnitude));
            }
        }
    }
    return worst;
}

} // namespace

TEST(OzakiContractionTest, Gemm_AlphaBetaTransposed) {
    auto ozaki = applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Modular, 0);
    EXPECT_LT(gemm_relative_error("ozaki_gemm", 2.0, 0.5, ozaki), 5e-16);
}

TEST(OzakiContractionTest, Gemm_BetaZeroIgnoresC) {
    auto ozaki = applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Slicing, 0);
    EXPECT_LT(gemm_relative_error("ozaki_gemm_beta0", 1.0, 0.0, ozaki), 5e-16);
}

TEST(QuantizeContractionTest, Gemm_AlphaBetaTransposed) {
    EXPECT_LT(gemm_relative_error("quantize_gemm", -1.5, 1.0, applying<fp_math::QuantizeContraction>()), 1e-2);
}

TEST(ContractionPrecisionTest, Parse) {
    using fp_math::ContractionPrecision;
    for (auto spec : {"native", "int8", "int4", "ozaki1", "ozaki2", "ozaki1:6", "ozaki2:12"}) {
        EXPECT_EQ(ContractionPrecision::parse(spec).str(), spec);
    }
    auto ozaki = ContractionPrecision::parse("ozaki");
    EXPECT_EQ(ozaki.kind, ContractionPrecision::Kind::Ozaki);
    EXPECT_EQ(ozaki.variant, fp_math::OzakiVariant::Modular);
    EXPECT_EQ(ozaki.parameter, 0);
    auto slicing = ContractionPrecision::parse("ozaki1:6");
    EXPECT_EQ(slicing.variant, fp_math::OzakiVariant::Slicing);
    EXPECT_EQ(slicing.parameter, 6);
    for (auto spec : {"", "int9", "int1", "ozaki3", "ozaki1:", "ozaki:4", "ozaki2:0", "fp16"}) {
        EXPECT_THROW(ContractionPrecision::parse(spec), std::invalid_argument) << spec;
    }
}

TEST(ContractionPrecisionTest, PassReplacesSupportedContractions) {
    builder::StructuredSDFGBuilder builder("precision_pass", FunctionType_CPU);
    auto& sdfg = builder.subject();
    add_gemm(builder, "1.0", "0.0");
    // A second, integer matmul that no precision applies to.
    types::Pointer int_pointer{types::Scalar(PrimitiveType::Int32)};
    builder.add_container("x", int_pointer, true);
    builder.add_container("w", int_pointer, true);
    builder.add_container("z", int_pointer, true);
    auto& block = builder.add_block(sdfg.root());
    auto& integer = builder.add_library_node<MatMulNode>(
        block,
        DebugInfo(),
        math::tensor::TensorLayout({symbolic::integer(M), symbolic::integer(K)}),
        math::tensor::TensorLayout({symbolic::integer(K), symbolic::integer(N)})
    );
    builder.add_computational_memlet(block, builder.add_access(block, "x"), integer, "A", {}, int_pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "w"), integer, "B", {}, int_pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "z"), integer, "Y", {}, int_pointer);

    analysis::AnalysisManager analysis_manager(sdfg);
    fp_math::ContractionPrecisionPass pass(fp_math::ContractionPrecision::parse("ozaki2:4"));
    EXPECT_TRUE(pass.run(builder, analysis_manager));
    EXPECT_NO_THROW(sdfg.validate());

    // The GEMM became 4 Int8 matmuls; the integer matmul stays.
    auto found = matmuls(sdfg.root());
    ASSERT_EQ(found.size(), 5);
    int int8 = 0;
    for (auto* matmul : found) {
        int8 += matmul->input_type() == PrimitiveType::Int8;
    }
    EXPECT_EQ(int8, 4);
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}

TEST(ContractionPrecisionTest, NativeChangesNothing) {
    builder::StructuredSDFGBuilder builder("precision_native", FunctionType_CPU);
    add_gemm(builder, "1.0", "0.0");
    analysis::AnalysisManager analysis_manager(builder.subject());
    fp_math::ContractionPrecisionPass pass(fp_math::ContractionPrecision::parse("native"));
    EXPECT_FALSE(pass.run(builder, analysis_manager));
}
