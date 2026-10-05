#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/plugin.h"
#include "sdfg/targets/gpu/math/tensor/tensor_operands.h"
#include "sdfg/targets/rocm/plugin.h"
#include "sdfg/targets/rocm/rocm.h"

using namespace sdfg;

namespace {

struct CUDABackend {
    static void register_plugin(plugins::Context& context) {
        cuda::register_cuda_plugin(context);
    }
    static const data_flow::ImplementationType& with_transfers() {
        return cuda::ImplementationType_CUDAWithTransfers;
    }
    static const data_flow::ImplementationType& without_transfers() {
        return cuda::ImplementationType_CUDAWithoutTransfers;
    }
    static constexpr const char* api = "cuda";
    static constexpr const char* launch = "<<<";
    static constexpr const char* kernel_ext = "cu";
};

struct ROCMBackend {
    static void register_plugin(plugins::Context& context) {
        rocm::register_rocm_plugin(context);
    }
    static const data_flow::ImplementationType& with_transfers() {
        return rocm::ImplementationType_ROCMWithTransfers;
    }
    static const data_flow::ImplementationType& without_transfers() {
        return rocm::ImplementationType_ROCMWithoutTransfers;
    }
    static constexpr const char* api = "hip";
    static constexpr const char* launch = "hipLaunchKernelGGL(";
    static constexpr const char* kernel_ext = "rocm.cpp";
};

size_t count(const std::string& haystack, const std::string& needle) {
    size_t n = 0;
    for (auto pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + 1)) {
        ++n;
    }
    return n;
}

struct Generated {
    std::string code;
    std::string globals;
    std::string kernel;
    std::string kernel_header;
};

template<typename Backend>
class GPUEmbeddingDispatcherTest : public ::testing::Test {
protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_ = make_builder();
    codegen::LibraryNodeDispatcherRegistry registry_;

    static std::unique_ptr<builder::StructuredSDFGBuilder> make_builder() {
        return std::make_unique<builder::StructuredSDFGBuilder>("embedding_dispatch", FunctionType_CPU);
    }

    void SetUp() override {
        plugins::Context context{
            serializer::LibraryNodeSerializerRegistry::instance(),
            codegen::NodeDispatcherRegistry::instance(),
            codegen::MapDispatcherRegistry::instance(),
            codegen::ReduceDispatcherRegistry::instance(),
            registry_,
            passes::scheduler::SchedulerRegistry::instance(),
            tiles::TileTargetRegistry::instance()
        };
        Backend::register_plugin(context);
    }

    math::tensor::EmbeddingNode& add_embedding(
        types::PrimitiveType weight_type,
        types::PrimitiveType index_type,
        const std::vector<symbolic::Expression>& weight_shape,
        const std::vector<symbolic::Expression>& index_shape
    ) {
        auto& sdfg = builder_->subject();
        types::Scalar weight_desc(weight_type);
        types::Scalar index_desc(index_type);
        builder_->add_container("W", types::Pointer(weight_desc), true);
        builder_->add_container("I", types::Pointer(index_desc), true);
        builder_->add_container("Y", types::Pointer(weight_desc), true);

        std::vector<symbolic::Expression> output_shape(index_shape.begin(), index_shape.end());
        output_shape.push_back(weight_shape.at(1));

        auto& block = builder_->add_block(sdfg.root());
        auto& w = builder_->add_access(block, "W");
        auto& i = builder_->add_access(block, "I");
        auto& y = builder_->add_access(block, "Y");
        auto& node = static_cast<
            math::tensor::EmbeddingNode&>(builder_->add_library_node<
                                          math::tensor::EmbeddingNode>(block, DebugInfo(), weight_shape, index_shape));
        builder_->add_computational_memlet(block, y, node, "Y", {}, types::Tensor(weight_desc, output_shape));
        builder_->add_computational_memlet(block, w, node, "W", {}, types::Tensor(weight_desc, weight_shape));
        builder_->add_computational_memlet(block, i, node, "I", {}, types::Tensor(index_desc, index_shape));
        return node;
    }

    math::tensor::EmbeddingNode& add_default_embedding() {
        return add_embedding(
            types::PrimitiveType::Float,
            types::PrimitiveType::Int64,
            {symbolic::integer(100), symbolic::integer(33)},
            {symbolic::integer(2), symbolic::integer(17)}
        );
    }

    Generated dispatch(math::tensor::EmbeddingNode& node, const data_flow::ImplementationType& impl_type) {
        node.set_implementation_type(impl_type);
        auto& sdfg = builder_->subject();
        sdfg.validate();

        auto dispatcher_fn = registry_.get_library_node_dispatcher(node.code().value() + "::" + impl_type.value());
        EXPECT_NE(dispatcher_fn, nullptr);
        if (!dispatcher_fn) {
            return {};
        }

        codegen::CLanguageExtension language_extension(sdfg);
        auto& dataflow = node.get_parent();
        auto dispatcher = dispatcher_fn(language_extension, sdfg, dataflow, node);

        codegen::PrettyPrinter stream;
        codegen::PrettyPrinter globals;
        codegen::CodeSnippetFactory snippets;
        dispatcher->dispatch(stream, globals, snippets);

        Generated generated{stream.str(), globals.str(), "", ""};
        for (auto& [name, snippet] : snippets.snippets()) {
            if (snippet.extension() == Backend::kernel_ext) {
                generated.kernel = snippet.stream().str();
            } else {
                generated.kernel_header = snippet.stream().str();
            }
        }
        return generated;
    }
};

using Backends = ::testing::Types<CUDABackend, ROCMBackend>;

} // namespace

TYPED_TEST_SUITE(GPUEmbeddingDispatcherTest, Backends);

TYPED_TEST(GPUEmbeddingDispatcherTest, RegisteredForBothTransferModes) {
    auto key = math::tensor::LibraryNodeType_Embedding.value() + "::";
    EXPECT_NE(this->registry_.get_library_node_dispatcher(key + TypeParam::with_transfers().value()), nullptr);
    EXPECT_NE(this->registry_.get_library_node_dispatcher(key + TypeParam::without_transfers().value()), nullptr);
}

TYPED_TEST(GPUEmbeddingDispatcherTest, WithTransfersCopiesOperands) {
    auto& node = this->add_default_embedding();
    auto generated = this->dispatch(node, TypeParam::with_transfers());
    std::string api = TypeParam::api;

    EXPECT_EQ(count(generated.code, api + "Malloc("), 3);
    // W and I are copied in, the write-only Y is only copied back.
    EXPECT_EQ(count(generated.code, api + "MemcpyHostToDevice"), 2);
    EXPECT_EQ(count(generated.code, api + "MemcpyDeviceToHost"), 1);
    EXPECT_EQ(count(generated.code, api + "Free("), 3);
    EXPECT_NE(generated.code.find(TypeParam::launch), std::string::npos);
    EXPECT_NE(generated.code.find("embedding_kernel_"), std::string::npos);

    auto launch = generated.code.find(TypeParam::launch);
    EXPECT_LT(generated.code.rfind("MemcpyHostToDevice", launch), launch);
    EXPECT_GT(generated.code.find("MemcpyDeviceToHost"), launch);
}

TYPED_TEST(GPUEmbeddingDispatcherTest, WithoutTransfersUsesDevicePointers) {
    auto& node = this->add_default_embedding();
    auto generated = this->dispatch(node, TypeParam::without_transfers());
    std::string api = TypeParam::api;

    EXPECT_EQ(generated.code.find(api + "Malloc("), std::string::npos);
    EXPECT_EQ(generated.code.find(api + "Memcpy("), std::string::npos);
    EXPECT_EQ(generated.code.find(api + "Free("), std::string::npos);
    EXPECT_NE(generated.code.find(TypeParam::launch), std::string::npos);
    for (auto* container : {"Y", "W", "I"}) {
        EXPECT_NE(generated.code.find(std::string("*) ") + container), std::string::npos) << container;
    }
}

TYPED_TEST(GPUEmbeddingDispatcherTest, EmitsGatherKernel) {
    auto& node = this->add_default_embedding();
    auto generated = this->dispatch(node, TypeParam::without_transfers());

    EXPECT_NE(generated.globals.find("__global__ void embedding_kernel_"), std::string::npos);
    EXPECT_NE(generated.kernel.find("__global__ void embedding_kernel_"), std::string::npos);
    EXPECT_NE(generated.kernel.find("out[e] = weight[(long long) indices[row] * dim + col];"), std::string::npos);
    EXPECT_NE(generated.kernel.find("uint32_t* __restrict__ out"), std::string::npos);
    EXPECT_NE(generated.kernel.find("const int64_t* __restrict__ indices"), std::string::npos);
    EXPECT_FALSE(generated.kernel_header.empty());
    // total = 2 * 17 * 33 output elements, dim = 33
    EXPECT_NE(generated.code.find("(long long) (1122)"), std::string::npos);
    EXPECT_NE(generated.code.find("(long long) (33)"), std::string::npos);
}

TYPED_TEST(GPUEmbeddingDispatcherTest, CopiesElementsByWidth) {
    struct Case {
        types::PrimitiveType weight;
        types::PrimitiveType index;
        const char* value_t;
        const char* index_t;
    };
    for (auto c :
         {Case{types::PrimitiveType::Half, types::PrimitiveType::Int32, "uint16_t", "int32_t"},
          Case{types::PrimitiveType::BFloat, types::PrimitiveType::Int64, "uint16_t", "int64_t"},
          Case{types::PrimitiveType::Double, types::PrimitiveType::Int32, "uint64_t", "int32_t"}}) {
        this->builder_ = this->make_builder();
        auto& node =
            this->add_embedding(c.weight, c.index, {symbolic::integer(10), symbolic::integer(4)}, {symbolic::integer(3)});
        auto generated = this->dispatch(node, TypeParam::without_transfers());
        EXPECT_NE(generated.kernel.find(std::string(c.value_t) + "* __restrict__ out"), std::string::npos)
            << types::primitive_type_to_string(c.weight);
        EXPECT_NE(generated.kernel.find(std::string("const ") + c.index_t + "* __restrict__ indices"), std::string::npos)
            << types::primitive_type_to_string(c.index);
    }
}

TYPED_TEST(GPUEmbeddingDispatcherTest, SymbolicShapes) {
    types::Scalar sym_desc(types::PrimitiveType::Int64);
    this->builder_->add_container("V", sym_desc, true);
    this->builder_->add_container("D", sym_desc, true);
    this->builder_->add_container("N", sym_desc, true);
    auto& node = this->add_embedding(
        types::PrimitiveType::Float,
        types::PrimitiveType::Int64,
        {symbolic::symbol("V"), symbolic::symbol("D")},
        {symbolic::symbol("N")}
    );
    auto generated = this->dispatch(node, TypeParam::with_transfers());

    auto operands = gpu::tensor::tensor_operands(node, node.get_parent());
    ASSERT_TRUE(operands.has_value());
    EXPECT_TRUE(
        symbolic::
            eq(gpu::tensor::find_operand(*operands, "Y").num_elements,
               symbolic::mul(symbolic::symbol("N"), symbolic::symbol("D")))
    );
    EXPECT_NE(generated.code.find("(long long) (D)"), std::string::npos);
}

TYPED_TEST(GPUEmbeddingDispatcherTest, RejectsUnsupportedIndexType) {
    auto& node = this->add_embedding(
        types::PrimitiveType::Float,
        types::PrimitiveType::Int16,
        {symbolic::integer(10), symbolic::integer(4)},
        {symbolic::integer(3)}
    );
    EXPECT_THROW(this->dispatch(node, TypeParam::without_transfers()), InvalidSDFGException);
}
