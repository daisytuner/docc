#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include <nlohmann/json.hpp>

#include "compile_and_load.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/copy_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/fp_math/library_nodes/ozaki_matmul_node.h"
#include "sdfg/fp_math/passes/contraction_precision.h"
#include "sdfg/fp_math/transformations/ozaki_contraction.h"
#include "sdfg/fp_math/transformations/quantize_contraction.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/transformations/transformation_registry.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/visitor/for_each.h"

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

/// Largest error of @p transform's result, relative to sum_k |A[i, k] B[k, j]|, with A scaled by @p a_scale.
template<typename T>
double relative_error(
    const std::string& name,
    const std::function<void(data_flow::LibraryNode&, builder::StructuredSDFGBuilder&)>& transform,
    double a_scale = 1.0
) {
    auto type = std::is_same_v<T, float> ? PrimitiveType::Float : PrimitiveType::Double;
    builder::StructuredSDFGBuilder builder(name, FunctionType_CPU);
    auto& matmul = add_matmul(builder, type, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    transform(matmul, builder);
    auto function = reinterpret_cast<void (*)(T*, T*, T*)>(test::compile_and_load(builder));

    std::vector<T> a, b, y(M * N, T(-7));
    fill(a, b);
    for (auto& x : a) {
        x = static_cast<T>(x * a_scale);
    }
    function(a.data(), b.data(), y.data());

    double worst = 0;
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            long double exact = 0, magnitude = 0;
            for (int64_t k = 0; k < K; k++) {
                exact += static_cast<long double>(a[i * K + k]) * b[k * N + j];
                magnitude += std::fabs(static_cast<long double>(a[i * K + k]) * b[k * N + j]);
            }
            // A NaN would slip through the maximum below.
            EXPECT_TRUE(std::isfinite(y[i * N + j])) << "row " << i << ", column " << j;
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

TEST(QuantizeContractionTest, ComputesInTheContractionsTypeWithMaxReductions) {
    // A Float matmul quantizes and scales in Float, so targets without Double run it; its scales are Max reductions.
    builder::StructuredSDFGBuilder builder("quantize_types", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Float, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    applying<fp_math::QuantizeContraction>()(matmul, builder);

    auto& sdfg = builder.subject();
    for (auto& name : sdfg.containers()) {
        EXPECT_NE(sdfg.type(name).primitive_type(), PrimitiveType::Double) << name;
    }
    size_t reductions = 0;
    std::function<void(structured_control_flow::ControlFlowNode&)> visit =
        [&](structured_control_flow::ControlFlowNode& node) {
            if (auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(&node)) {
                ASSERT_EQ(reduce->reductions().size(), 1);
                EXPECT_EQ(reduce->reductions().front().operation, structured_control_flow::ReductionOperation::Max);
                ++reductions;
            }
            if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
                for (size_t i = 0; i < sequence->size(); i++) {
                    visit(sequence->at(i));
                }
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
                visit(loop->root());
            }
        };
    visit(sdfg.root());
    EXPECT_EQ(reductions, 2);
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

TEST(QuantizeContractionTest, Double_WithinQuantizationError) {
    EXPECT_LT(relative_error<double>("quantize_double", applying<fp_math::QuantizeContraction>()), 1e-2);
}

TEST(QuantizeContractionTest, TinyRowsDoNotOverflow) {
    // Rows whose largest magnitude lies below q / max_finite, where the factor q / s would overflow to infinity.
    EXPECT_LT(relative_error<double>("quantize_double_tiny", applying<fp_math::QuantizeContraction>(), 1e-303), 1e-2);
    EXPECT_LT(relative_error<float>("quantize_float_tiny", applying<fp_math::QuantizeContraction>(), 1e-34), 1e-2);
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
    nlohmann::json direct;
    fp_math::QuantizeContraction(matmul, 8, fp_math::Rescaling::Direct).to_json(direct);
    EXPECT_EQ(direct["parameters"]["rescaling"], "direct");
    EXPECT_TRUE(transformations::verify_transformation_round_trip(builder, direct, error)) << error;
}

TEST(QuantizeContractionTest, Direct_MultipliesOnly) {
    // Direct rescaling needs no floor and no powers of two, for units that compute neither exactly.
    builder::StructuredSDFGBuilder builder("quantize_direct_structure", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Float, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    applying<fp_math::QuantizeContraction>(8, fp_math::Rescaling::Direct)(matmul, builder);
    std::function<void(structured_control_flow::ControlFlowNode&)> visit =
        [&](structured_control_flow::ControlFlowNode& node) {
            if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
                for (auto* library_node : block->dataflow().library_nodes()) {
                    auto* cmath = dynamic_cast<math::cmath::CMathNode*>(library_node);
                    EXPECT_TRUE(
                        !cmath || (cmath->function() != math::cmath::CMathFunction::floor &&
                                   cmath->function() != math::cmath::CMathFunction::exp2)
                    );
                }
            } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
                for (size_t i = 0; i < sequence->size(); i++) {
                    visit(sequence->at(i));
                }
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
                visit(loop->root());
            }
        };
    visit(builder.subject().root());
}

TEST(QuantizeContractionTest, Direct_WithinQuantizationError) {
    auto direct = applying<fp_math::QuantizeContraction>(8, fp_math::Rescaling::Direct);
    EXPECT_LT(relative_error<float>("quantize_float_direct", direct), 1e-2);
}

namespace {

constexpr int64_t H = 3;

/// Largest error of the quantized Y[h] = A[h] @ B[h] over H matrices relative to sum_k |A[h, i, k] B[h, k, j]|, where
/// matrix h of A is scaled by 2^(20 h), so that scales shared across the batch would lose the smaller ones. B is stored
/// transposed per matrix, as the keys of attention scores are, or is one matrix all of them share if @p shared_b.
double batched_relative_error(
    const std::function<void(data_flow::LibraryNode&, builder::StructuredSDFGBuilder&)>& transform,
    bool shared_b,
    size_t int8_matmuls = 1
) {
    builder::StructuredSDFGBuilder builder(shared_b ? "quantize_broadcast" : "quantize_batched", FunctionType_CPU);
    types::Pointer pointer{types::Scalar(PrimitiveType::Float)};
    for (auto name : {"a", "b", "y"}) {
        builder.add_container(name, pointer, true);
    }
    auto& block = builder.add_block(builder.subject().root());
    using math::tensor::TensorLayout;
    auto i = [](int64_t value) {
        return symbolic::integer(value);
    };
    TensorLayout layout_a({i(H), i(M), i(K)});
    TensorLayout layout_b = shared_b ? TensorLayout({i(K), i(N)})
                                     : TensorLayout({i(H), i(K), i(N)}, {i(K * N), i(1), i(K)});
    TensorLayout layout_y({i(H), i(M), i(N)});
    auto& matmul = static_cast<MatMulNode&>(builder.add_library_node<MatMulNode>(
        block, DebugInfo(), layout_a, layout_b, math::tensor::QUANTIZATION_MATCH_INPUTS, &layout_y
    ));
    builder.add_computational_memlet(block, builder.add_access(block, "a"), matmul, "A", {}, pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "b"), matmul, "B", {}, pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "y"), matmul, "Y", {}, pointer);
    transform(matmul, builder);

    auto found = matmuls(builder.subject().root());
    EXPECT_EQ(found.size(), int8_matmuls);
    for (auto* int8 : found) {
        EXPECT_EQ(int8->layout_a().dims(), 3);
        EXPECT_EQ(int8->layout_b().dims(), 3);
    }
    auto function = reinterpret_cast<void (*)(float*, float*, float*)>(test::compile_and_load(builder));

    std::vector<float> a, b, y(H * M * N, -7.0f);
    std::vector<float> a_matrix, b_matrix;
    fill(a_matrix, b_matrix);
    if (shared_b) {
        b = b_matrix;
    }
    for (int64_t h = 0; h < H; h++) {
        for (auto x : a_matrix) {
            a.push_back(std::ldexp(x, static_cast<int>(20 * h)));
        }
        // Stored transposed: element [k, j] of matrix h at h K N + j K + k.
        for (int64_t j = 0; j < N && !shared_b; j++) {
            for (int64_t k = 0; k < K; k++) {
                b.push_back(b_matrix[k * N + j] * static_cast<float>(h + 1));
            }
        }
    }
    function(a.data(), b.data(), y.data());

    double worst = 0;
    for (int64_t h = 0; h < H; h++) {
        for (int64_t row = 0; row < M; row++) {
            for (int64_t column = 0; column < N; column++) {
                long double exact = 0, magnitude = 0;
                for (int64_t k = 0; k < K; k++) {
                    auto b_value = shared_b ? b[k * N + column] : b[h * K * N + column * K + k];
                    auto product = static_cast<long double>(a[h * M * K + row * K + k]) * b_value;
                    exact += product;
                    magnitude += std::fabs(product);
                }
                auto value = y[h * M * N + row * N + column];
                EXPECT_TRUE(std::isfinite(value)) << h << ", " << row << ", " << column;
                if (magnitude != 0) {
                    worst = std::max(worst, static_cast<double>(std::fabs(value - exact) / magnitude));
                }
            }
        }
    }
    return worst;
}

} // namespace

TEST(QuantizeContractionTest, Batched_ScalesEachMatrix) {
    for (auto rescaling : {fp_math::Rescaling::Exponents, fp_math::Rescaling::Direct}) {
        EXPECT_LT(batched_relative_error(applying<fp_math::QuantizeContraction>(8, rescaling), false), 1e-2);
    }
}

TEST(QuantizeContractionTest, Batched_BroadcastsASharedOperand) {
    EXPECT_LT(batched_relative_error(applying<fp_math::QuantizeContraction>(8, fp_math::Rescaling::Direct), true), 1e-2);
}

TEST(OzakiContractionTest, RejectsBatchedMatMuls) {
    builder::StructuredSDFGBuilder builder("ozaki_batched", FunctionType_CPU);
    types::Pointer pointer{types::Scalar(PrimitiveType::Double)};
    for (auto name : {"a", "b", "y"}) {
        builder.add_container(name, pointer, true);
    }
    auto& block = builder.add_block(builder.subject().root());
    auto shape = [](int64_t rows, int64_t columns) {
        return math::tensor::TensorLayout({symbolic::integer(H), symbolic::integer(rows), symbolic::integer(columns)});
    };
    auto& matmul = builder.add_library_node<MatMulNode>(block, DebugInfo(), shape(M, K), shape(K, N));
    builder.add_computational_memlet(block, builder.add_access(block, "a"), matmul, "A", {}, pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "b"), matmul, "B", {}, pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "y"), matmul, "Y", {}, pointer);
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_FALSE(fp_math::OzakiContraction(matmul).can_be_applied(builder, analysis_manager));
    EXPECT_TRUE(fp_math::QuantizeContraction(matmul).can_be_applied(builder, analysis_manager));
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
    for (auto [variant, rescaling] :
         {std::pair{fp_math::OzakiVariant::Slicing, fp_math::Rescaling::Exponents},
          std::pair{fp_math::OzakiVariant::Modular, fp_math::Rescaling::Exponents},
          std::pair{fp_math::OzakiVariant::Slicing, fp_math::Rescaling::Direct}}) {
        nlohmann::json description;
        fp_math::OzakiContraction(matmul, variant, 3, rescaling).to_json(description);
        std::string error;
        EXPECT_TRUE(transformations::verify_transformation_round_trip(builder, description, error)) << error;
    }
}

namespace {

auto ozaki_direct(int slices) {
    return applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Slicing, slices, fp_math::Rescaling::Direct);
}

} // namespace

TEST(OzakiContractionTest, Direct_SlicesInTheContractionsTypeByRounding) {
    // For units without Double, floor or exact powers of two: rint, the absolute maxima's fabs and fmax, and
    // multiplications only, and s(s+1)/2 Int8 matmuls.
    builder::StructuredSDFGBuilder builder("ozaki_direct_structure", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Float, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    ozaki_direct(2)(matmul, builder);

    auto& sdfg = builder.subject();
    for (auto& name : sdfg.containers()) {
        EXPECT_NE(sdfg.type(name).primitive_type(), PrimitiveType::Double) << name;
    }
    EXPECT_EQ(matmuls(sdfg.root()).size(), 3);
    std::function<void(structured_control_flow::ControlFlowNode&)> visit =
        [&](structured_control_flow::ControlFlowNode& node) {
            if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
                for (auto* library_node : block->dataflow().library_nodes()) {
                    if (auto* cmath = dynamic_cast<math::cmath::CMathNode*>(library_node)) {
                        using math::cmath::CMathFunction;
                        auto function = cmath->function();
                        EXPECT_TRUE(
                            function == CMathFunction::rint || function == CMathFunction::fabs ||
                            function == CMathFunction::fmax
                        ) << cmath->name();
                    }
                }
            } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
                for (size_t i = 0; i < sequence->size(); i++) {
                    visit(sequence->at(i));
                }
            } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
                visit(loop->root());
            }
        };
    visit(sdfg.root());
}

TEST(OzakiContractionTest, Direct_TwoSlicesBetweenInt8AndFloat) {
    // Two slices keep about 15 bits of each row and column, far beyond int8 quantization, far below Float.
    auto error = relative_error<float>("ozaki_direct_2", ozaki_direct(2));
    EXPECT_LT(error, 1e-4);
    EXPECT_GT(error, 1e-7);
}

TEST(OzakiContractionTest, Direct_Float_AtFloatPrecision) {
    EXPECT_LT(relative_error<float>("ozaki_direct_float", ozaki_direct(0)), 1e-6);
}

TEST(OzakiContractionTest, Direct_Batched_ScalesEachMatrix) {
    EXPECT_LT(batched_relative_error(ozaki_direct(2), false, 3), 1e-4);
}

namespace {

/// B's slices and the absolute maxima of its columns, as slicing with direct rescaling prepares them.
std::pair<std::vector<std::vector<int8_t>>, std::vector<float>> prepare_b(const std::vector<float>& b, int slices) {
    const float q = 127.0f;
    std::vector<float> scales(N, std::numeric_limits<float>::min());
    for (int64_t k = 0; k < K; k++) {
        for (int64_t j = 0; j < N; j++) {
            scales[j] = std::max(scales[j], std::fabs(b[k * N + j]));
        }
    }
    std::vector<std::vector<int8_t>> digits(slices, std::vector<int8_t>(K * N));
    for (int64_t e = 0; e < K * N; e++) {
        float rest = b[e] / scales[e % N] * q;
        for (int t = 0; t < slices; t++) {
            if (t > 0) {
                rest = (rest - digits[t - 1][e]) * (2 * q);
            }
            digits[t][e] = static_cast<int8_t>(std::rint(rest));
        }
    }
    return {digits, scales};
}

/// y = a @ b through an OzakiMatMul node, with b's two slices prepared by prepare_b.
std::vector<float> prepared_product(const std::vector<float>& a, const std::vector<float>& b) {
    builder::StructuredSDFGBuilder builder("ozaki_matmul_prepared", FunctionType_CPU);
    auto& sdfg = builder.subject();
    types::Pointer f32{types::Scalar(PrimitiveType::Float)};
    types::Pointer i8{types::Scalar(PrimitiveType::Int8)};
    for (auto name : {"a", "sb", "y"}) {
        builder.add_container(name, f32, true);
    }
    builder.add_container("b0", i8, true);
    builder.add_container("b1", i8, true);
    auto& block = builder.add_block(sdfg.root());
    auto& node = builder.add_library_node<fp_math::OzakiMatMulNode>(
        block,
        DebugInfo(),
        math::tensor::TensorLayout({symbolic::integer(M), symbolic::integer(K)}),
        math::tensor::TensorLayout({symbolic::integer(M), symbolic::integer(N)}),
        PrimitiveType::Float,
        2
    );
    for (auto [container, connector] : {std::pair{"y", "Y"}, {"a", "A"}, {"sb", "SB"}, {"b0", "B0"}, {"b1", "B1"}}) {
        auto& type = container[0] == 'b' ? i8 : f32;
        builder.add_computational_memlet(block, builder.add_access(block, container), node, connector, {}, type);
    }

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(fp_math::expand_ozaki_matmuls(builder, analysis_manager));
    EXPECT_EQ(matmuls(sdfg.root()).size(), 3);
    auto function =
        reinterpret_cast<void (*)(float*, float*, float*, int8_t*, int8_t*)>(test::compile_and_load(builder));
    auto [digits, scales] = prepare_b(b, 2);
    std::vector<float> a_copy = a, y(M * N, -7.0f);
    function(a_copy.data(), scales.data(), y.data(), digits[0].data(), digits[1].data());
    return y;
}

} // namespace

TEST(OzakiContractionTest, Prepared_MatchesSlicingInTheGraph) {
    // B sliced before, once, computes what slicing it with A computes.
    std::vector<float> a, b;
    fill(a, b);
    auto prepared = prepared_product(a, b);

    builder::StructuredSDFGBuilder builder("ozaki_direct_reference", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Float, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    ozaki_direct(2)(matmul, builder);
    auto function = reinterpret_cast<void (*)(float*, float*, float*)>(test::compile_and_load(builder));
    std::vector<float> direct(M * N, -7.0f);
    function(a.data(), b.data(), direct.data());

    for (int64_t e = 0; e < M * N; e++) {
        EXPECT_NEAR(prepared[e], direct[e], 1e-6f * std::max(1.0f, std::fabs(direct[e]))) << "element " << e;
    }
}

TEST(OzakiContractionTest, Prepared_JsonRoundTrip) {
    builder::StructuredSDFGBuilder builder("ozaki_matmul_json", FunctionType_CPU);
    auto& block = builder.add_block(builder.subject().root());
    auto& node = static_cast<fp_math::OzakiMatMulNode&>(builder.add_library_node<fp_math::OzakiMatMulNode>(
        block,
        DebugInfo(),
        math::tensor::TensorLayout({symbolic::integer(M), symbolic::integer(K)}),
        math::tensor::TensorLayout({symbolic::integer(M), symbolic::integer(N)}),
        PrimitiveType::Float,
        3
    ));
    auto json = fp_math::OzakiMatMulNodeSerializer().serialize(node);
    json["debug_info"] = nlohmann::json::object();
    serializer::JSONSerializer().debug_info_to_json(json["debug_info"], DebugInfo());
    auto& restored =
        static_cast<fp_math::OzakiMatMulNode&>(fp_math::OzakiMatMulNodeSerializer().deserialize(json, builder, block));
    EXPECT_EQ(restored.slices(), 3);
    EXPECT_EQ(restored.inputs().size(), 6);
    EXPECT_TRUE(symbolic::eq(restored.n(), symbolic::integer(N)));
    EXPECT_TRUE(symbolic::eq(restored.k(), symbolic::integer(K)));
}

TEST(OzakiContractionTest, Direct_RejectsModular) {
    builder::StructuredSDFGBuilder builder("ozaki_direct_modular", FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Float, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    analysis::AnalysisManager analysis_manager(builder.subject());
    fp_math::OzakiContraction modular(matmul, fp_math::OzakiVariant::Modular, 0, fp_math::Rescaling::Direct);
    EXPECT_FALSE(modular.can_be_applied(builder, analysis_manager));
}

namespace {

/// C = alpha A @ B^T + beta C over the argument pointers a[M, K], b[N, K], c[M, N] of Double, as a GEMM node; beta
/// "runtime" reads beta from an argument instead of a constant.
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
    if (beta == "runtime") {
        builder.add_container("beta", element, true);
        builder.add_computational_memlet(block, builder.add_access(block, "beta"), gemm, "__beta", {}, element);
    } else {
        builder.add_computational_memlet(block, builder.add_constant(block, beta, element), gemm, "__beta", {}, element);
    }
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
            EXPECT_TRUE(std::isfinite(c[i * N + j])) << "row " << i << ", column " << j;
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
    auto direct = applying<fp_math::QuantizeContraction>(8, fp_math::Rescaling::Direct);
    EXPECT_LT(gemm_relative_error("quantize_gemm_direct", -1.5, 1.0, direct), 1e-2);
}

TEST(QuantizeContractionTest, Gemm_QuantizesInStorageOrder) {
    // B is stored transposed (b[N, K]): its quantization reads along its rows and keeps it so, and the Int8 matmul
    // takes B in that order.
    builder::StructuredSDFGBuilder builder("quantize_storage", FunctionType_CPU);
    auto& gemm = add_gemm(builder, "1.0", "0.0");
    applying<fp_math::QuantizeContraction>()(gemm, builder);

    auto found = matmuls(builder.subject().root());
    ASSERT_EQ(found.size(), 1);
    auto& layout_b = found.front()->layout_b();
    EXPECT_EQ(layout_b.is_2d_col_or_row_major(), math::tensor::TensorLayout::LAYOUT_COL_MAJOR);
    EXPECT_TRUE(symbolic::eq(layout_b.get_stride(1), symbolic::integer(K)));
    EXPECT_EQ(found.front()->layout_a().is_2d_col_or_row_major(), math::tensor::TensorLayout::LAYOUT_ROW_MAJOR);
}

TEST(OzakiContractionTest, Gemm_DirectTransposed) {
    EXPECT_LT(gemm_relative_error("ozaki_gemm_direct", 2.0, 0.5, ozaki_direct(2)), 1e-4);
}

TEST(ContractionTest, Direct_ReadsRowMajorCopyOfTransposedB) {
    // B is stored transposed (b[N, K]): a TensorCopy makes it row-major first, so that the scales of its columns
    // reduce along rows and every Int8 matmul takes B row-major.
    for (auto transform : {applying<fp_math::QuantizeContraction>(8, fp_math::Rescaling::Direct), ozaki_direct(2)}) {
        builder::StructuredSDFGBuilder builder("direct_row_major_b", FunctionType_CPU);
        auto& gemm = add_gemm(builder, "1.0", "0.0");
        transform(gemm, builder);

        int copies = 0;
        visitor::for_each_block(builder.subject().root(), [&](structured_control_flow::Block& block) {
            for (auto* node : block.dataflow().library_nodes()) {
                copies += dynamic_cast<math::tensor::TensorCopyNode*>(node) != nullptr;
            }
        });
        EXPECT_EQ(copies, 1);
        auto found = matmuls(builder.subject().root());
        ASSERT_FALSE(found.empty());
        for (auto* matmul : found) {
            EXPECT_EQ(matmul->layout_b().is_2d_col_or_row_major(), math::tensor::TensorLayout::LAYOUT_ROW_MAJOR);
        }
    }
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

TEST(OzakiContractionTest, Gemm_RequiresConstantCoefficients) {
    // BLAS skips C for beta = 0 and A @ B for alpha = 0, which only constants reveal.
    auto applies = [](const std::string& alpha, const std::string& beta) {
        builder::StructuredSDFGBuilder builder("gemm_coefficients", FunctionType_CPU);
        auto& gemm = add_gemm(builder, alpha, beta);
        analysis::AnalysisManager analysis_manager(builder.subject());
        bool quantize = fp_math::QuantizeContraction(gemm).can_be_applied(builder, analysis_manager);
        bool ozaki = fp_math::OzakiContraction(gemm).can_be_applied(builder, analysis_manager);
        EXPECT_EQ(quantize, ozaki);
        return ozaki;
    };
    EXPECT_TRUE(applies("2.0", "0.5"));
    EXPECT_FALSE(applies("1.0", "runtime"));
    EXPECT_FALSE(applies("0.0", "1.0"));
}

namespace {

/// Y = A @ B for A = 2^1022 and B = 2^-1022 everywhere: Y = K exactly, although 2^1022 * K overflows.
void expect_extreme_exponents_exact(
    const std::string& name,
    const std::function<void(data_flow::LibraryNode&, builder::StructuredSDFGBuilder&)>& transform
) {
    builder::StructuredSDFGBuilder builder(name, FunctionType_CPU);
    auto& matmul =
        add_matmul(builder, PrimitiveType::Double, symbolic::integer(M), symbolic::integer(N), symbolic::integer(K));
    transform(matmul, builder);
    auto function = reinterpret_cast<void (*)(double*, double*, double*)>(test::compile_and_load(builder));
    std::vector<double> a(M * K, std::ldexp(1.0, 1022)), b(K * N, std::ldexp(1.0, -1022)), y(M * N, -1.0);
    function(a.data(), b.data(), y.data());
    for (auto value : y) {
        EXPECT_EQ(value, static_cast<double>(K)) << name;
    }
}

} // namespace

TEST(OzakiContractionTest, ExtremeExponentsDoNotOverflow) {
    expect_extreme_exponents_exact(
        "ozaki_extreme_slicing", applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Slicing, 0)
    );
    expect_extreme_exponents_exact(
        "ozaki_extreme_modular", applying<fp_math::OzakiContraction>(fp_math::OzakiVariant::Modular, 0)
    );
}

TEST(QuantizeContractionTest, ExtremeExponentsDoNotOverflow) {
    expect_extreme_exponents_exact("quantize_extreme", applying<fp_math::QuantizeContraction>());
}
