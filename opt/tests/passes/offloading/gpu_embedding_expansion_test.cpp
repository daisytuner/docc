#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/passes/dataflow/tensor_to_pointer_conversion.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/passes/offloading/cuda_library_node_expansion_pass.h"
#include "sdfg/passes/offloading/rocm_library_node_expansion_pass.h"
#include "sdfg/structured_control_flow/map.h"

using namespace sdfg;

namespace {

struct CUDABackend {
    using Expansion = passes::CudaExpansionPass;
};

struct ROCMBackend {
    using Expansion = passes::RocmExpansionPass;
};

template<typename Backend>
class GPUEmbeddingExpansionTest : public ::testing::Test {
protected:
    builder::StructuredSDFGBuilder builder_{"embedding_expansion", FunctionType_CPU};
    structured_control_flow::Block* block_ = nullptr;

    void SetUp() override {
        auto& sdfg = builder_.subject();
        types::Scalar float_desc(types::PrimitiveType::Float);
        types::Scalar index_desc(types::PrimitiveType::Int64);
        builder_.add_container("W", types::Pointer(float_desc), true);
        builder_.add_container("I", types::Pointer(index_desc), true);
        builder_.add_container("Y", types::Pointer(float_desc), true);

        std::vector<symbolic::Expression> weight_shape = {symbolic::integer(10), symbolic::integer(4)};
        std::vector<symbolic::Expression> index_shape = {symbolic::integer(3)};
        std::vector<symbolic::Expression> output_shape = {symbolic::integer(3), symbolic::integer(4)};

        block_ = &builder_.add_block(sdfg.root());
        auto& w = builder_.add_access(*block_, "W");
        auto& i = builder_.add_access(*block_, "I");
        auto& y = builder_.add_access(*block_, "Y");
        auto& node =
            builder_.add_library_node<math::tensor::EmbeddingNode>(*block_, DebugInfo(), weight_shape, index_shape);
        builder_.add_computational_memlet(*block_, y, node, "Y", {}, types::Tensor(float_desc, output_shape));
        builder_.add_computational_memlet(*block_, w, node, "W", {}, types::Tensor(float_desc, weight_shape));
        builder_.add_computational_memlet(*block_, i, node, "I", {}, types::Tensor(index_desc, index_shape));
    }

    math::tensor::EmbeddingNode* embedding() {
        auto library_nodes = block_->dataflow().library_nodes();
        if (library_nodes.size() != 1) {
            return nullptr;
        }
        return dynamic_cast<math::tensor::EmbeddingNode*>(*library_nodes.begin());
    }
};

using Backends = ::testing::Types<CUDABackend, ROCMBackend>;

} // namespace

TYPED_TEST_SUITE(GPUEmbeddingExpansionTest, Backends);

TYPED_TEST(GPUEmbeddingExpansionTest, LeftToGenericExpansion) {
    auto& sdfg = this->builder_.subject();
    analysis::AnalysisManager analysis_manager(sdfg);

    typename TypeParam::Expansion target_expansion;
    EXPECT_FALSE(target_expansion.run(this->builder_, analysis_manager));
    ASSERT_NE(this->embedding(), nullptr);
    EXPECT_EQ(this->embedding()->implementation_type().value(), data_flow::ImplementationType_NONE.value());

    passes::LibraryNodeExpansionPass generic_expansion;
    EXPECT_TRUE(generic_expansion.run(this->builder_, analysis_manager));
    passes::TensorToPointerConversionPass tensor_to_pointer;
    tensor_to_pointer.run(this->builder_, analysis_manager);
    EXPECT_NO_THROW(sdfg.validate());
    ASSERT_GT(sdfg.root().size(), 0);
    EXPECT_NE(dynamic_cast<structured_control_flow::Map*>(&sdfg.root().at(0)), nullptr);
}

TYPED_TEST(GPUEmbeddingExpansionTest, LeavesAssignedImplementationUntouched) {
    data_flow::ImplementationType preset("Preset");
    this->embedding()->set_implementation_type(preset);

    analysis::AnalysisManager analysis_manager(this->builder_.subject());
    typename TypeParam::Expansion target_expansion;
    EXPECT_FALSE(target_expansion.run(this->builder_, analysis_manager));
    EXPECT_EQ(this->embedding()->implementation_type().value(), preset.value());
}
