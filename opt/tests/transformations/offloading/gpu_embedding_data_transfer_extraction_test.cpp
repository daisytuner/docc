#include "sdfg/transformations/offloading/gpu_embedding_data_transfer_extraction.h"

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/passes/offloading/cuda_library_node_transfer_extraction_pass.h"
#include "sdfg/passes/offloading/rocm_library_node_transfer_extraction_pass.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/plugin.h"
#include "sdfg/targets/rocm/plugin.h"
#include "sdfg/targets/rocm/rocm.h"

using namespace sdfg;

namespace {

struct CUDABackend {
    using Extraction = gpu::tensor::CUDAEmbeddingDataTransferExtraction;
    using Pass = cuda::CudaLibraryNodeTransferExtractionPass;
    static void register_plugin(plugins::Context& context) {
        cuda::register_cuda_plugin(context);
    }
    static const data_flow::ImplementationType& with_transfers() {
        return cuda::ImplementationType_CUDAWithTransfers;
    }
    static const data_flow::ImplementationType& without_transfers() {
        return cuda::ImplementationType_CUDAWithoutTransfers;
    }
    static const data_flow::ImplementationType& other_with_transfers() {
        return rocm::ImplementationType_ROCMWithTransfers;
    }
    static const std::string& device_prefix() {
        return cuda::CUDA_DEVICE_PREFIX;
    }
    static constexpr const char* api = "cuda";
    static constexpr const char* storage = "NV_Generic";
};

struct ROCMBackend {
    using Extraction = gpu::tensor::ROCMEmbeddingDataTransferExtraction;
    using Pass = rocm::RocmLibraryNodeTransferExtractionPass;
    static void register_plugin(plugins::Context& context) {
        rocm::register_rocm_plugin(context);
    }
    static const data_flow::ImplementationType& with_transfers() {
        return rocm::ImplementationType_ROCMWithTransfers;
    }
    static const data_flow::ImplementationType& without_transfers() {
        return rocm::ImplementationType_ROCMWithoutTransfers;
    }
    static const data_flow::ImplementationType& other_with_transfers() {
        return cuda::ImplementationType_CUDAWithTransfers;
    }
    static const std::string& device_prefix() {
        return rocm::ROCM_DEVICE_PREFIX;
    }
    static constexpr const char* api = "hip";
    static constexpr const char* storage = "AMD_Generic";
};

struct Transfer {
    offloading::DataTransferDirection direction;
    offloading::BufferLifecycle lifecycle;
    std::string host;
    symbolic::Expression size;
};

template<typename Backend>
class GPUEmbeddingDataTransferExtractionTest : public ::testing::Test {
protected:
    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;
    math::tensor::EmbeddingNode* node_ = nullptr;

    void SetUp() override {
        build("Y");
    }

    // Y is bound to `y_container`; binding it to "W" makes the gather in-place.
    void build(const std::string& y_container) {
        builder_ = std::make_unique<builder::StructuredSDFGBuilder>("embedding_extraction", FunctionType_CPU);
        auto& sdfg = builder_->subject();
        types::Scalar float_desc(types::PrimitiveType::Float);
        types::Scalar index_desc(types::PrimitiveType::Int32);
        builder_->add_container("W", types::Pointer(float_desc), true);
        builder_->add_container("I", types::Pointer(index_desc), true);
        if (y_container != "W") {
            builder_->add_container(y_container, types::Pointer(float_desc), true);
        }

        std::vector<symbolic::Expression> weight_shape = {symbolic::integer(100), symbolic::integer(33)};
        std::vector<symbolic::Expression> index_shape = {symbolic::integer(2), symbolic::integer(17)};
        std::vector<symbolic::Expression> output_shape = {
            symbolic::integer(2), symbolic::integer(17), symbolic::integer(33)
        };

        auto& block = builder_->add_block(sdfg.root());
        auto& w = builder_->add_access(block, "W");
        auto& i = builder_->add_access(block, "I");
        auto& y = y_container == "W" ? w : builder_->add_access(block, y_container);
        node_ = &static_cast<
            math::tensor::EmbeddingNode&>(builder_->add_library_node<
                                          math::tensor::EmbeddingNode>(block, DebugInfo(), weight_shape, index_shape));
        builder_->add_computational_memlet(block, y, *node_, "Y", {}, types::Tensor(float_desc, output_shape));
        builder_->add_computational_memlet(block, w, *node_, "W", {}, types::Tensor(float_desc, weight_shape));
        builder_->add_computational_memlet(block, i, *node_, "I", {}, types::Tensor(index_desc, index_shape));
    }

    bool run_pass() {
        analysis::AnalysisManager analysis_manager(builder_->subject());
        typename Backend::Pass pass;
        return pass.run(*builder_, analysis_manager);
    }

    std::string connected_container(const std::string& connector) {
        auto* edge = node_->get_parent().in_edge_for_connector(*node_, connector);
        return static_cast<const data_flow::AccessNode&>(edge->src()).data();
    }

    std::vector<Transfer> transfers() {
        std::vector<Transfer> result;
        auto& root = builder_->subject().root();
        for (size_t i = 0; i < root.size(); ++i) {
            auto* block = dynamic_cast<structured_control_flow::Block*>(&root.at(i));
            if (block == nullptr) {
                continue;
            }
            auto& dfg = block->dataflow();
            for (auto* lib_node : dfg.library_nodes()) {
                auto* offload = dynamic_cast<offloading::DataOffloadingNode*>(lib_node);
                if (offload == nullptr) {
                    continue;
                }
                Transfer transfer{offload->transfer_direction(), offload->buffer_lifecycle(), "", offload->size()};
                if (auto* edge = dfg.in_edge_for_connector(*offload, "_hst")) {
                    transfer.host = static_cast<const data_flow::AccessNode&>(edge->src()).data();
                }
                result.push_back(transfer);
            }
        }
        return result;
    }

    std::string dispatch_code() {
        codegen::LibraryNodeDispatcherRegistry registry;
        plugins::Context context{
            serializer::LibraryNodeSerializerRegistry::instance(),
            codegen::NodeDispatcherRegistry::instance(),
            codegen::MapDispatcherRegistry::instance(),
            codegen::ReduceDispatcherRegistry::instance(),
            registry,
            passes::scheduler::SchedulerRegistry::instance(),
            tiles::TileTargetRegistry::instance()
        };
        Backend::register_plugin(context);
        auto dispatcher_fn =
            registry.get_library_node_dispatcher(node_->code().value() + "::" + node_->implementation_type().value());
        EXPECT_NE(dispatcher_fn, nullptr);
        if (!dispatcher_fn) {
            return "";
        }

        auto& sdfg = builder_->subject();
        codegen::CLanguageExtension language_extension(sdfg);
        auto dispatcher = dispatcher_fn(language_extension, sdfg, node_->get_parent(), *node_);
        codegen::PrettyPrinter stream;
        codegen::PrettyPrinter globals;
        codegen::CodeSnippetFactory snippets;
        dispatcher->dispatch(stream, globals, snippets);
        return stream.str();
    }
};

using Backends = ::testing::Types<CUDABackend, ROCMBackend>;

} // namespace

TYPED_TEST_SUITE(GPUEmbeddingDataTransferExtractionTest, Backends);

TYPED_TEST(GPUEmbeddingDataTransferExtractionTest, ExtractsEmbeddingTransfers) {
    this->node_->set_implementation_type(TypeParam::with_transfers());

    EXPECT_TRUE(this->run_pass());
    auto& sdfg = this->builder_->subject();
    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(this->node_->implementation_type().value(), TypeParam::without_transfers().value());
    // alloc/copy-in of W, I, Y + node + copy-out/free of W, I, Y
    EXPECT_EQ(sdfg.root().size(), 7);

    for (auto* connector : {"Y", "W", "I"}) {
        auto container = this->connected_container(connector);
        EXPECT_EQ(container.rfind(TypeParam::device_prefix(), 0), 0u) << connector;
        EXPECT_EQ(sdfg.type(container).storage_type().value(), TypeParam::storage) << connector;
    }
    EXPECT_EQ(sdfg.type(this->connected_container("I")).primitive_type(), types::PrimitiveType::Int32);

    std::map<std::string, std::pair<int, int>> copies;
    int allocs = 0;
    int frees = 0;
    for (auto& transfer : this->transfers()) {
        if (transfer.direction == offloading::DataTransferDirection::H2D) {
            copies[transfer.host].first++;
        }
        if (transfer.direction == offloading::DataTransferDirection::D2H) {
            copies[transfer.host].second++;
        }
        allocs += transfer.lifecycle == offloading::BufferLifecycle::ALLOC;
        frees += transfer.lifecycle == offloading::BufferLifecycle::FREE;
        if (transfer.host == "W") {
            EXPECT_TRUE(symbolic::eq(transfer.size, symbolic::integer(100 * 33 * 4)));
        }
        if (transfer.host == "I") {
            EXPECT_TRUE(symbolic::eq(transfer.size, symbolic::integer(2 * 17 * 4)));
        }
        if (transfer.host == "Y") {
            EXPECT_TRUE(symbolic::eq(transfer.size, symbolic::integer(2 * 17 * 33 * 4)));
        }
    }
    EXPECT_EQ(allocs, 3);
    EXPECT_EQ(frees, 3);
    EXPECT_EQ(copies["W"], std::make_pair(1, 0));
    EXPECT_EQ(copies["I"], std::make_pair(1, 0));
    // Y is fully overwritten: allocated without a copy-in, copied back once.
    EXPECT_EQ(copies["Y"], std::make_pair(0, 1));
}

TYPED_TEST(GPUEmbeddingDataTransferExtractionTest, SkipsAliasedContainers) {
    this->build("W");
    this->node_->set_implementation_type(TypeParam::with_transfers());

    EXPECT_FALSE(this->run_pass());
    EXPECT_EQ(this->node_->implementation_type().value(), TypeParam::with_transfers().value());
}

TYPED_TEST(GPUEmbeddingDataTransferExtractionTest, ExtractedNodeDispatchesWithoutTransfers) {
    this->node_->set_implementation_type(TypeParam::with_transfers());
    ASSERT_TRUE(this->run_pass());

    auto code = this->dispatch_code();
    EXPECT_EQ(code.find(std::string(TypeParam::api) + "Malloc("), std::string::npos);
    EXPECT_NE(code.find(this->connected_container("Y")), std::string::npos);
}

TYPED_TEST(GPUEmbeddingDataTransferExtractionTest, SkipsOtherImplementationTypes) {
    for (auto& impl_type : {data_flow::ImplementationType_NONE, TypeParam::other_with_transfers()}) {
        this->node_->set_implementation_type(impl_type);
        EXPECT_FALSE(this->run_pass()) << impl_type.value();
        EXPECT_EQ(this->builder_->subject().root().size(), 1);
        EXPECT_EQ(this->connected_container("W"), "W");
    }
}

TYPED_TEST(GPUEmbeddingDataTransferExtractionTest, SkipsNonIsolatedNode) {
    this->node_->set_implementation_type(TypeParam::with_transfers());

    auto& builder = *this->builder_;
    auto& block = static_cast<structured_control_flow::Block&>(*this->node_->get_parent().get_parent());
    types::Scalar float_desc(types::PrimitiveType::Float);
    builder.add_container("x", float_desc);
    builder.add_container("z", float_desc);
    auto& x = builder.add_access(block, "x");
    auto& z = builder.add_access(block, "z");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, x, tasklet, "_in", {}, float_desc);
    builder.add_computational_memlet(block, tasklet, "_out", z, {}, float_desc);

    analysis::AnalysisManager analysis_manager(builder.subject());
    typename TypeParam::Extraction extraction(*this->node_);
    EXPECT_FALSE(extraction.can_be_applied(builder, analysis_manager));
}
