#include "sdfg/passes/memory/tensor_allocation_size_inference.h"

#include <gtest/gtest.h>

#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

using namespace sdfg;

namespace {

const types::Scalar float_desc(types::PrimitiveType::Float);
const types::Scalar index_desc(types::PrimitiveType::Int64);

// EmbeddingNode only serves as a library node consuming three Tensor-typed memlets.
void add_tensor_use(
    builder::StructuredSDFGBuilder& builder,
    const types::Tensor& w_type,
    const types::Tensor& i_type,
    const types::Tensor& y_type
) {
    auto& block = builder.add_block(builder.subject().root());
    auto& w = builder.add_access(block, "W");
    auto& i = builder.add_access(block, "I");
    auto& y = builder.add_access(block, "Y");
    auto& node =
        builder.add_library_node<math::tensor::EmbeddingNode>(block, DebugInfo(), w_type.shape(), i_type.shape());
    builder.add_computational_memlet(block, y, node, "Y", {}, y_type);
    builder.add_computational_memlet(block, w, node, "W", {}, w_type);
    builder.add_computational_memlet(block, i, node, "I", {}, i_type);
}

void add_containers(builder::StructuredSDFGBuilder& builder) {
    builder.add_container("W", types::Pointer(float_desc), true);
    builder.add_container("I", types::Pointer(index_desc), true);
    builder.add_container("Y", types::Pointer(float_desc));
}

types::Tensor tensor(const types::Scalar& element, const symbolic::MultiExpression& shape) {
    return types::Tensor(element, math::tensor::TensorLayout(shape));
}

symbolic::Expression allocation_size(builder::StructuredSDFGBuilder& builder, const std::string& container) {
    return builder.subject().type(container).storage_type().allocation_size();
}

bool run_pass(builder::StructuredSDFGBuilder& builder) {
    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::TensorAllocationSizeInference pass;
    return pass.run(builder, analysis_manager);
}

} // namespace

TEST(TensorAllocationSizeInferenceTest, ContiguousTensors) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_containers(builder);
    add_tensor_use(
        builder,
        tensor(float_desc, {symbolic::integer(10), symbolic::integer(4)}),
        tensor(index_desc, {symbolic::integer(3)}),
        tensor(float_desc, {symbolic::integer(3), symbolic::integer(4)})
    );

    EXPECT_TRUE(run_pass(builder));

    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "W"), symbolic::integer(160)));
    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "I"), symbolic::integer(24)));
    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "Y"), symbolic::integer(48)));
    for (auto& container : {"W", "I", "Y"}) {
        auto storage = builder.subject().type(container).storage_type();
        EXPECT_EQ(storage.allocation(), types::StorageType::Unmanaged);
        EXPECT_EQ(storage.deallocation(), types::StorageType::Unmanaged);
    }
}

TEST(TensorAllocationSizeInferenceTest, StridedLayoutIncludesGaps) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_containers(builder);
    math::tensor::TensorLayout strided(
        {symbolic::integer(10), symbolic::integer(4)},
        {symbolic::integer(8), symbolic::integer(1)},
        symbolic::integer(2)
    );
    add_tensor_use(
        builder,
        types::Tensor(float_desc, strided),
        tensor(index_desc, {symbolic::integer(3)}),
        tensor(float_desc, {symbolic::integer(3), symbolic::integer(4)})
    );

    EXPECT_TRUE(run_pass(builder));

    // (2 + 9 * 8 + 3 * 1 + 1) elements * 4 bytes
    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "W"), symbolic::integer(312)));
}

TEST(TensorAllocationSizeInferenceTest, SymbolicArgumentShape) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_containers(builder);
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);
    auto N = symbolic::symbol("N");
    add_tensor_use(
        builder,
        tensor(float_desc, {N, symbolic::integer(4)}),
        tensor(index_desc, {symbolic::integer(3)}),
        tensor(float_desc, {symbolic::integer(3), symbolic::integer(4)})
    );

    EXPECT_TRUE(run_pass(builder));

    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "W"), symbolic::mul(symbolic::integer(16), N)));
}

TEST(TensorAllocationSizeInferenceTest, NonArgumentSymbolRejectsContainer) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_containers(builder);
    builder.add_container("M", types::Scalar(types::PrimitiveType::Int64));
    add_tensor_use(
        builder,
        tensor(float_desc, {symbolic::symbol("M"), symbolic::integer(4)}),
        tensor(index_desc, {symbolic::integer(3)}),
        tensor(float_desc, {symbolic::integer(3), symbolic::integer(4)})
    );
    add_tensor_use(
        builder,
        tensor(float_desc, {symbolic::integer(10), symbolic::integer(4)}),
        tensor(index_desc, {symbolic::integer(3)}),
        tensor(float_desc, {symbolic::integer(3), symbolic::integer(4)})
    );

    EXPECT_TRUE(run_pass(builder));

    EXPECT_TRUE(allocation_size(builder, "W").is_null());
    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "I"), symbolic::integer(24)));
}

TEST(TensorAllocationSizeInferenceTest, MultipleUsesTakeMaximum) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_containers(builder);
    add_tensor_use(
        builder,
        tensor(float_desc, {symbolic::integer(10), symbolic::integer(4)}),
        tensor(index_desc, {symbolic::integer(3)}),
        tensor(float_desc, {symbolic::integer(3), symbolic::integer(4)})
    );
    add_tensor_use(
        builder,
        tensor(float_desc, {symbolic::integer(20), symbolic::integer(4)}),
        tensor(index_desc, {symbolic::integer(2)}),
        tensor(float_desc, {symbolic::integer(2), symbolic::integer(4)})
    );

    EXPECT_TRUE(run_pass(builder));

    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "W"), symbolic::integer(320)));
    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "I"), symbolic::integer(24)));
    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "Y"), symbolic::integer(48)));
}

TEST(TensorAllocationSizeInferenceTest, ManagedAllocationUntouched) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    types::Pointer managed(
        types::StorageType::CPU_Heap(symbolic::integer(1024), types::StorageType::Managed, types::StorageType::Managed),
        0,
        "",
        float_desc
    );
    builder.add_container("W", managed);
    builder.add_container("I", types::Pointer(index_desc), true);
    builder.add_container("Y", types::Pointer(float_desc), true);
    add_tensor_use(
        builder,
        tensor(float_desc, {symbolic::integer(10), symbolic::integer(4)}),
        tensor(index_desc, {symbolic::integer(3)}),
        tensor(float_desc, {symbolic::integer(3), symbolic::integer(4)})
    );

    EXPECT_TRUE(run_pass(builder));

    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "W"), symbolic::integer(1024)));
}

TEST(TensorAllocationSizeInferenceTest, Idempotent) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_containers(builder);
    add_tensor_use(
        builder,
        tensor(float_desc, {symbolic::integer(10), symbolic::integer(4)}),
        tensor(index_desc, {symbolic::integer(3)}),
        tensor(float_desc, {symbolic::integer(3), symbolic::integer(4)})
    );

    EXPECT_TRUE(run_pass(builder));
    EXPECT_FALSE(run_pass(builder));

    EXPECT_TRUE(symbolic::eq(allocation_size(builder, "W"), symbolic::integer(160)));
}

TEST(TensorAllocationSizeInferenceTest, SurvivesSerialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    add_containers(builder);
    builder.add_container("N", types::Scalar(types::PrimitiveType::Int64), true);
    add_tensor_use(
        builder,
        tensor(float_desc, {symbolic::symbol("N"), symbolic::integer(4)}),
        tensor(index_desc, {symbolic::integer(3)}),
        tensor(float_desc, {symbolic::integer(3), symbolic::integer(4)})
    );
    EXPECT_TRUE(run_pass(builder));

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto copy = serializer.deserialize(j);

    EXPECT_TRUE(symbolic::eq(copy->type("W").storage_type().allocation_size(), allocation_size(builder, "W")));
}
