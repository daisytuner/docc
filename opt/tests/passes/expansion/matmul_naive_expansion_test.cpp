#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "compile_and_load.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;
using types::PrimitiveType;

namespace {

constexpr int64_t M = 3, N = 4, K = 5;

/// Y[M, N] = A[M, K] @ B[K, N], with inputs of type @p input and the output (and accumulator) of type @p output.
void* compile_mixed_matmul(const std::string& name, PrimitiveType input, PrimitiveType output) {
    builder::StructuredSDFGBuilder builder(name, FunctionType_CPU);
    auto& sdfg = builder.subject();
    types::Pointer input_pointer{types::Scalar(input)};
    types::Pointer output_pointer{types::Scalar(output)};
    builder.add_container("a", input_pointer, true);
    builder.add_container("b", input_pointer, true);
    builder.add_container("y", output_pointer, true);
    auto& block = builder.add_block(sdfg.root());
    auto m = symbolic::integer(M), n = symbolic::integer(N), k = symbolic::integer(K);
    auto& matmul = builder.add_library_node<math::tensor::MatMulNode>(
        block,
        DebugInfo(),
        math::tensor::TensorLayout({m, k}),
        math::tensor::TensorLayout({k, n}),
        math::tensor::ContractionTypes{input, PrimitiveType::Void, output}
    );
    builder.add_computational_memlet(block, builder.add_access(block, "a"), matmul, "A", {}, input_pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "b"), matmul, "B", {}, input_pointer);
    builder.add_computational_memlet(block, builder.add_access(block, "y"), matmul, "Y", {}, output_pointer);
    return test::compile_and_load(builder);
}

} // namespace

TEST(MatMulNaiveExpansionTest, FloatInputsMultiplyInDouble) {
    // Every product 1e20 * 1e20 overflows Float, but not the Double accumulator.
    auto function = reinterpret_cast<void (*)(float*, float*, double*)>(
        compile_mixed_matmul("naive_float_double", PrimitiveType::Float, PrimitiveType::Double)
    );
    std::vector<float> a(M * K, 1e20f), b(K * N, 1e20f);
    std::vector<double> y(M * N, -1.0);
    function(a.data(), b.data(), y.data());
    double expected = K * static_cast<double>(1e20f) * static_cast<double>(1e20f);
    for (auto value : y) {
        EXPECT_DOUBLE_EQ(value, expected);
    }
}

TEST(MatMulNaiveExpansionTest, Int32InputsMultiplyInInt64) {
    // Every product 50000 * 50000 overflows Int32, but not the Int64 accumulator.
    auto function = reinterpret_cast<void (*)(int32_t*, int32_t*, int64_t*)>(
        compile_mixed_matmul("naive_int32_int64", PrimitiveType::Int32, PrimitiveType::Int64)
    );
    std::vector<int32_t> a(M * K, 50000), b(K * N, 50000);
    std::vector<int64_t> y(M * N, -1);
    function(a.data(), b.data(), y.data());
    for (auto value : y) {
        EXPECT_EQ(value, K * int64_t{50000} * 50000);
    }
}
