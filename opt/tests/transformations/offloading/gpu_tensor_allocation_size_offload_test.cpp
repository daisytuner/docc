#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/passes/dataflow/tensor_to_pointer_conversion.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/passes/memory/tensor_allocation_size_inference.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/transformations/offloading/cuda_offload_transform.h"
#include "sdfg/transformations/offloading/rocm_offload_transform.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

using namespace sdfg;

namespace {

structured_control_flow::Map* find_outer_map(structured_control_flow::ControlFlowNode& node) {
    if (auto* map = dynamic_cast<structured_control_flow::Map*>(&node)) {
        return map;
    }
    if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            if (auto* map = find_outer_map(sequence->at(i))) {
                return map;
            }
        }
    }
    return nullptr;
}

std::map<std::string, symbolic::Expression> h2d_sizes(builder::StructuredSDFGBuilder& builder) {
    std::map<std::string, symbolic::Expression> sizes;
    auto& root = builder.subject().root();
    for (size_t i = 0; i < root.size(); ++i) {
        auto* block = dynamic_cast<structured_control_flow::Block*>(&root.at(i));
        if (block == nullptr) {
            continue;
        }
        auto& dataflow = block->dataflow();
        for (auto* lib_node : dataflow.library_nodes()) {
            auto* offloading = dynamic_cast<offloading::DataOffloadingNode*>(lib_node);
            if (offloading == nullptr || !offloading->is_h2d()) {
                continue;
            }
            for (auto& edge : dataflow.in_edges(*offloading)) {
                if (edge.dst_conn() != offloading->host_in_conn()) {
                    continue;
                }
                auto& host = static_cast<const data_flow::AccessNode&>(edge.src());
                sizes.insert({host.data(), offloading->size()});
            }
        }
    }
    return sizes;
}

template<typename Transform>
class GPUTensorAllocationSizeOffloadTest : public ::testing::Test {
protected:
    builder::StructuredSDFGBuilder builder_{"embedding_offload", FunctionType_CPU};

    void SetUp() override {
        types::Scalar float_desc(types::PrimitiveType::Float);
        types::Scalar index_desc(types::PrimitiveType::Int64);
        builder_.add_container("W", types::Pointer(float_desc), true);
        builder_.add_container("I", types::Pointer(index_desc), true);
        builder_.add_container("Y", types::Pointer(float_desc), true);

        std::vector<symbolic::Expression> weight_shape = {symbolic::integer(10), symbolic::integer(4)};
        std::vector<symbolic::Expression> index_shape = {symbolic::integer(3)};
        std::vector<symbolic::Expression> output_shape = {symbolic::integer(3), symbolic::integer(4)};

        auto& block = builder_.add_block(builder_.subject().root());
        auto& w = builder_.add_access(block, "W");
        auto& i = builder_.add_access(block, "I");
        auto& y = builder_.add_access(block, "Y");
        auto& node =
            builder_.add_library_node<math::tensor::EmbeddingNode>(block, DebugInfo(), weight_shape, index_shape);
        builder_.add_computational_memlet(block, y, node, "Y", {}, types::Tensor(float_desc, output_shape));
        builder_.add_computational_memlet(block, w, node, "W", {}, types::Tensor(float_desc, weight_shape));
        builder_.add_computational_memlet(block, i, node, "I", {}, types::Tensor(index_desc, index_shape));
    }

    structured_control_flow::Map& expand(analysis::AnalysisManager& analysis_manager, bool infer_sizes) {
        if (infer_sizes) {
            passes::TensorAllocationSizeInference inference;
            inference.run(builder_, analysis_manager);
        }
        passes::LibraryNodeExpansionPass expansion;
        expansion.run(builder_, analysis_manager);
        passes::TensorToPointerConversionPass tensor_to_pointer;
        tensor_to_pointer.run(builder_, analysis_manager);

        auto* map = find_outer_map(builder_.subject().root());
        EXPECT_NE(map, nullptr);
        return *map;
    }
};

using Transforms = ::testing::Types<cuda::CUDAOffloadTransform, rocm::ROCMOffloadTransform>;

} // namespace

TYPED_TEST_SUITE(GPUTensorAllocationSizeOffloadTest, Transforms);

TYPED_TEST(GPUTensorAllocationSizeOffloadTest, GatherNotOffloadableWithoutSizes) {
    analysis::AnalysisManager analysis_manager(this->builder_.subject());
    auto& map = this->expand(analysis_manager, false);

    TypeParam transform(map, symbolic::integer(32), gpu::TargetLevel::X_GRID);
    EXPECT_FALSE(transform.can_be_applied(this->builder_, analysis_manager));
}

TYPED_TEST(GPUTensorAllocationSizeOffloadTest, GatherOffloadsWithTensorSizes) {
    analysis::AnalysisManager analysis_manager(this->builder_.subject());
    auto& map = this->expand(analysis_manager, true);

    TypeParam transform(map, symbolic::integer(32), gpu::TargetLevel::X_GRID);
    ASSERT_TRUE(transform.can_be_applied(this->builder_, analysis_manager));
    transform.apply(this->builder_, analysis_manager);
    analysis_manager.invalidate_all();
    EXPECT_NO_THROW(this->builder_.subject().validate());

    auto sizes = h2d_sizes(this->builder_);
    ASSERT_TRUE(sizes.contains("W"));
    EXPECT_TRUE(symbolic::eq(sizes.at("W"), symbolic::integer(160)));
    ASSERT_TRUE(sizes.contains("I"));
    EXPECT_TRUE(symbolic::eq(sizes.at("I"), symbolic::integer(24)));
}
