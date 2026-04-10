#include "sdfg/passes/dataflow/local_buffer_reuse.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/relu_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

using namespace sdfg;

namespace {

/**
 * Helper to create the 6-block pattern for LocalBufferReuse:
 * 1. malloc1 -> ptr1
 * 2. ptr1 -> ref1 (reference)
 * 3. T library node -> ref1
 * 4. malloc2 -> ptr2
 * 5. ptr2 -> ref2 (reference)
 * 6. S library node: ref1 -> ref2
 * 7. free ptr1
 * 8. free ptr2
 *
 * T and S are template parameters for the library nodes involved.
 */
template<class T_first, class T_second>
class LocalBufferReuseTestSetup {
public:
    builder::StructuredSDFGBuilder builder;
    types::Scalar element_desc;
    types::Pointer ptr_desc;
    symbolic::MultiExpression tensor_shape;
    types::Tensor tensor_type;
    symbolic::Expression malloc_size;

    LocalBufferReuseTestSetup()
        : builder("sdfg_tensor_elim", FunctionType_CPU), element_desc(types::PrimitiveType::Float),
          ptr_desc(element_desc),
          tensor_shape({symbolic::integer(1), symbolic::integer(64), symbolic::integer(32), symbolic::integer(32)}),
          tensor_type(element_desc, tensor_shape), malloc_size(symbolic::integer(1 * 64 * 32 * 32 * 4)) // 4 bytes per
                                                                                                        // float
    {}

    void setup_containers() {
        builder.add_container("ptr1", ptr_desc);
        builder.add_container("ref1", ptr_desc);
        builder.add_container("ptr2", ptr_desc);
        builder.add_container("ref2", ptr_desc);
    }

    structured_control_flow::Block& add_malloc_block(const std::string& output_container, const symbolic::Expression& size) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);
        auto& access = builder.add_access(block, output_container);
        auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), size);
        builder.add_computational_memlet(block, malloc_node, "_ret", access, {}, ptr_desc, DebugInfo());
        return block;
    }

    structured_control_flow::Block& add_reference_block(const std::string& src_container, const std::string& dst_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);
        auto& src_access = builder.add_access(block, src_container);
        auto& dst_access = builder.add_access(block, dst_container);
        builder.add_reference_memlet(block, src_access, dst_access, {}, ptr_desc);
        return block;
    }

    structured_control_flow::Block& add_free_block(const std::string& container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);
        auto& access_in = builder.add_access(block, container);
        auto& access_out = builder.add_access(block, container);
        auto& free_node = builder.add_library_node<stdlib::FreeNode>(block, DebugInfo());
        builder.add_computational_memlet(block, access_in, free_node, "_ptr", {}, ptr_desc, DebugInfo());
        builder.add_computational_memlet(block, free_node, "_ptr", access_out, {}, ptr_desc, DebugInfo());
        return block;
    }
};

// Specialization for Conv -> BatchNorm elimination
class ConvBatchNormSetup : public LocalBufferReuseTestSetup<math::tensor::ConvNode, math::tensor::BatchNormNode> {
public:
    structured_control_flow::Block& add_conv_block(const std::string& output_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);

        // Create input containers for conv
        builder.add_container("conv_input", ptr_desc, true);
        builder.add_container("conv_weights", ptr_desc, true);
        auto& input_access = builder.add_access(block, "conv_input");
        auto& weights_access = builder.add_access(block, "conv_weights");
        auto& output_access = builder.add_access(block, output_container);

        std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
        std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
        std::vector<symbolic::Expression> pads = {
            symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
        };
        std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
        auto group = symbolic::integer(1);
        auto output_channels = symbolic::integer(64);

        auto& conv_node = builder.add_library_node<math::tensor::ConvNode>(
            block,
            DebugInfo(),
            std::vector<symbolic::Expression>(tensor_shape.begin(), tensor_shape.end()),
            kernel_shape,
            strides,
            pads,
            dilations,
            output_channels,
            group
        );

        types::Tensor weights_type(
            element_desc,
            std::vector<symbolic::Expression>{
                symbolic::integer(64), symbolic::integer(64), symbolic::integer(3), symbolic::integer(3)
            }
        );

        builder.add_computational_memlet(block, input_access, conv_node, "X", {}, tensor_type, DebugInfo());
        builder.add_computational_memlet(block, weights_access, conv_node, "W", {}, weights_type, DebugInfo());
        builder.add_computational_memlet(block, conv_node, "Y", output_access, {}, tensor_type, DebugInfo());

        return block;
    }

    structured_control_flow::Block&
    add_batchnorm_block(const std::string& input_container, const std::string& output_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);

        // Create weight containers for batchnorm
        builder.add_container("bn_var", ptr_desc, true);
        builder.add_container("bn_mean", ptr_desc, true);
        builder.add_container("bn_gamma", ptr_desc, true);
        builder.add_container("bn_beta", ptr_desc, true);

        auto& input_access = builder.add_access(block, input_container);
        auto& var_access = builder.add_access(block, "bn_var");
        auto& mean_access = builder.add_access(block, "bn_mean");
        auto& gamma_access = builder.add_access(block, "bn_gamma");
        auto& beta_access = builder.add_access(block, "bn_beta");
        auto& output_access = builder.add_access(block, output_container);
        auto& epsilon_node = builder.add_constant(block, "0.00001", element_desc);

        types::Tensor norm_type(element_desc, std::vector<symbolic::Expression>{symbolic::integer(64)});

        auto& bn_node = builder.add_library_node<math::tensor::BatchNormNode>(
            block,
            DebugInfo(),
            math::tensor::TensorLayout(std::vector<symbolic::Expression>(tensor_shape.begin(), tensor_shape.end())),
            types::Float
        );
        builder.add_computational_memlet(block, input_access, bn_node, "Batch", {}, tensor_type, DebugInfo());
        builder.add_computational_memlet(block, var_access, bn_node, "Var", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, mean_access, bn_node, "E", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, gamma_access, bn_node, "Gamma", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, beta_access, bn_node, "Beta", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, epsilon_node, bn_node, "epsilon", {}, element_desc, DebugInfo());
        // B_out is an INPUT connector (in-place destination buffer)
        builder.add_computational_memlet(block, output_access, bn_node, "B_out", {}, tensor_type, DebugInfo());

        return block;
    }
};

// Specialization for BatchNorm -> ReLU elimination
class BatchNormReLUSetup : public LocalBufferReuseTestSetup<math::tensor::BatchNormNode, math::tensor::ReLUNode> {
public:
    structured_control_flow::Block& add_batchnorm_block(const std::string& output_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);

        builder.add_container("bn_input", ptr_desc, true);
        builder.add_container("bn_var", ptr_desc, true);
        builder.add_container("bn_mean", ptr_desc, true);
        builder.add_container("bn_gamma", ptr_desc, true);
        builder.add_container("bn_beta", ptr_desc, true);

        auto& input_access = builder.add_access(block, "bn_input");
        auto& var_access = builder.add_access(block, "bn_var");
        auto& mean_access = builder.add_access(block, "bn_mean");
        auto& gamma_access = builder.add_access(block, "bn_gamma");
        auto& beta_access = builder.add_access(block, "bn_beta");
        auto& output_access = builder.add_access(block, output_container);
        auto& epsilon_node = builder.add_constant(block, "0.00001", element_desc);

        types::Tensor norm_type(element_desc, std::vector<symbolic::Expression>{symbolic::integer(64)});

        auto& bn_node = builder.add_library_node<math::tensor::BatchNormNode>(
            block,
            DebugInfo(),
            math::tensor::TensorLayout(std::vector<symbolic::Expression>(tensor_shape.begin(), tensor_shape.end())),
            types::Float
        );

        builder.add_computational_memlet(block, input_access, bn_node, "Batch", {}, tensor_type, DebugInfo());
        builder.add_computational_memlet(block, var_access, bn_node, "Var", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, mean_access, bn_node, "E", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, gamma_access, bn_node, "Gamma", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, beta_access, bn_node, "Beta", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, epsilon_node, bn_node, "epsilon", {}, element_desc, DebugInfo());
        // B_out is an INPUT connector (in-place destination buffer)
        builder.add_computational_memlet(block, output_access, bn_node, "B_out", {}, tensor_type, DebugInfo());

        return block;
    }

    structured_control_flow::Block& add_relu_block(const std::string& input_container, const std::string& output_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);

        auto& input_access = builder.add_access(block, input_container);
        auto& output_access = builder.add_access(block, output_container);

        auto& relu_node = builder.add_library_node<math::tensor::ReLUNode>(
            block, DebugInfo(), std::vector<symbolic::Expression>(tensor_shape.begin(), tensor_shape.end())
        );

        builder.add_computational_memlet(block, input_access, relu_node, "X", {}, tensor_type, DebugInfo());
        builder.add_computational_memlet(block, relu_node, "Y", output_access, {}, tensor_type, DebugInfo());

        return block;
    }
};

// Specialization for 5-node chain: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
class ConvBatchNormReLUConvBatchNormSetup {
public:
    builder::StructuredSDFGBuilder builder;
    types::Scalar element_desc;
    types::Pointer ptr_desc;
    symbolic::MultiExpression tensor_shape;
    types::Tensor tensor_type;
    types::Tensor norm_type;
    types::Tensor weights_type;
    symbolic::Expression malloc_size;
    int conv_counter = 0;
    int bn_counter = 0;

    ConvBatchNormReLUConvBatchNormSetup()
        : builder("sdfg_tensor_elim_5", FunctionType_CPU), element_desc(types::PrimitiveType::Float),
          ptr_desc(element_desc),
          tensor_shape({symbolic::integer(1), symbolic::integer(64), symbolic::integer(32), symbolic::integer(32)}),
          tensor_type(element_desc, tensor_shape),
          norm_type(element_desc, std::vector<symbolic::Expression>{symbolic::integer(64)}),
          weights_type(
              element_desc,
              std::vector<symbolic::Expression>{
                  symbolic::integer(64), symbolic::integer(64), symbolic::integer(3), symbolic::integer(3)
              }
          ),
          malloc_size(symbolic::integer(1 * 64 * 32 * 32 * 4)) {}

    void setup_containers() {
        for (int i = 1; i <= 5; i++) {
            builder.add_container("ptr" + std::to_string(i), ptr_desc);
            builder.add_container("ref" + std::to_string(i), ptr_desc);
        }
    }

    structured_control_flow::Block& add_malloc_block(const std::string& output_container, const symbolic::Expression& size) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);
        auto& access = builder.add_access(block, output_container);
        auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), size);
        builder.add_computational_memlet(block, malloc_node, "_ret", access, {}, ptr_desc, DebugInfo());
        return block;
    }

    structured_control_flow::Block& add_reference_block(const std::string& src_container, const std::string& dst_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);
        auto& src_access = builder.add_access(block, src_container);
        auto& dst_access = builder.add_access(block, dst_container);
        builder.add_reference_memlet(block, src_access, dst_access, {}, ptr_desc);
        return block;
    }

    structured_control_flow::Block& add_free_block(const std::string& container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);
        auto& access_in = builder.add_access(block, container);
        auto& access_out = builder.add_access(block, container);
        auto& free_node = builder.add_library_node<stdlib::FreeNode>(block, DebugInfo());
        builder.add_computational_memlet(block, access_in, free_node, "_ptr", {}, ptr_desc, DebugInfo());
        builder.add_computational_memlet(block, free_node, "_ptr", access_out, {}, ptr_desc, DebugInfo());
        return block;
    }

    structured_control_flow::Block& add_conv_block(const std::string& input_container, const std::string& output_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);
        conv_counter++;

        std::string weights_name = "conv_weights_" + std::to_string(conv_counter);
        builder.add_container(weights_name, ptr_desc, true);

        auto& input_access = builder.add_access(block, input_container);
        auto& weights_access = builder.add_access(block, weights_name);
        auto& output_access = builder.add_access(block, output_container);

        std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
        std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
        std::vector<symbolic::Expression> pads = {
            symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
        };
        std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
        auto group = symbolic::integer(1);
        auto output_channels = symbolic::integer(64);

        auto& conv_node = builder.add_library_node<math::tensor::ConvNode>(
            block,
            DebugInfo(),
            std::vector<symbolic::Expression>(tensor_shape.begin(), tensor_shape.end()),
            kernel_shape,
            strides,
            pads,
            dilations,
            output_channels,
            group
        );

        builder.add_computational_memlet(block, input_access, conv_node, "X", {}, tensor_type, DebugInfo());
        builder.add_computational_memlet(block, weights_access, conv_node, "W", {}, weights_type, DebugInfo());
        builder.add_computational_memlet(block, conv_node, "Y", output_access, {}, tensor_type, DebugInfo());

        return block;
    }

    structured_control_flow::Block&
    add_batchnorm_block(const std::string& input_container, const std::string& output_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);
        bn_counter++;

        std::string var_name = "bn_var_" + std::to_string(bn_counter);
        std::string mean_name = "bn_mean_" + std::to_string(bn_counter);
        std::string gamma_name = "bn_gamma_" + std::to_string(bn_counter);
        std::string beta_name = "bn_beta_" + std::to_string(bn_counter);

        builder.add_container(var_name, ptr_desc, true);
        builder.add_container(mean_name, ptr_desc, true);
        builder.add_container(gamma_name, ptr_desc, true);
        builder.add_container(beta_name, ptr_desc, true);

        auto& input_access = builder.add_access(block, input_container);
        auto& var_access = builder.add_access(block, var_name);
        auto& mean_access = builder.add_access(block, mean_name);
        auto& gamma_access = builder.add_access(block, gamma_name);
        auto& beta_access = builder.add_access(block, beta_name);
        auto& output_access = builder.add_access(block, output_container);
        auto& epsilon_node = builder.add_constant(block, "0.00001", element_desc);

        auto& bn_node = builder.add_library_node<math::tensor::BatchNormNode>(
            block,
            DebugInfo(),
            math::tensor::TensorLayout(std::vector<symbolic::Expression>(tensor_shape.begin(), tensor_shape.end())),
            types::Float
        );

        builder.add_computational_memlet(block, input_access, bn_node, "Batch", {}, tensor_type, DebugInfo());
        builder.add_computational_memlet(block, var_access, bn_node, "Var", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, mean_access, bn_node, "E", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, gamma_access, bn_node, "Gamma", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, beta_access, bn_node, "Beta", {}, norm_type, DebugInfo());
        builder.add_computational_memlet(block, epsilon_node, bn_node, "epsilon", {}, element_desc, DebugInfo());
        builder.add_computational_memlet(block, output_access, bn_node, "B_out", {}, tensor_type, DebugInfo());

        return block;
    }

    structured_control_flow::Block& add_relu_block(const std::string& input_container, const std::string& output_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);

        auto& input_access = builder.add_access(block, input_container);
        auto& output_access = builder.add_access(block, output_container);

        auto& relu_node = builder.add_library_node<math::tensor::ReLUNode>(
            block, DebugInfo(), std::vector<symbolic::Expression>(tensor_shape.begin(), tensor_shape.end())
        );

        builder.add_computational_memlet(block, input_access, relu_node, "X", {}, tensor_type, DebugInfo());
        builder.add_computational_memlet(block, relu_node, "Y", output_access, {}, tensor_type, DebugInfo());

        return block;
    }

    // First conv block (has external input)
    structured_control_flow::Block& add_first_conv_block(const std::string& output_container) {
        auto& root = builder.subject().root();
        auto& block = builder.add_block(root);
        conv_counter++;

        builder.add_container("conv_input", ptr_desc, true);
        std::string weights_name = "conv_weights_" + std::to_string(conv_counter);
        builder.add_container(weights_name, ptr_desc, true);

        auto& input_access = builder.add_access(block, "conv_input");
        auto& weights_access = builder.add_access(block, weights_name);
        auto& output_access = builder.add_access(block, output_container);

        std::vector<symbolic::Expression> kernel_shape = {symbolic::integer(3), symbolic::integer(3)};
        std::vector<symbolic::Expression> strides = {symbolic::integer(1), symbolic::integer(1)};
        std::vector<symbolic::Expression> pads = {
            symbolic::integer(1), symbolic::integer(1), symbolic::integer(1), symbolic::integer(1)
        };
        std::vector<symbolic::Expression> dilations = {symbolic::integer(1), symbolic::integer(1)};
        auto group = symbolic::integer(1);
        auto output_channels = symbolic::integer(64);

        auto& conv_node = builder.add_library_node<math::tensor::ConvNode>(
            block,
            DebugInfo(),
            std::vector<symbolic::Expression>(tensor_shape.begin(), tensor_shape.end()),
            kernel_shape,
            strides,
            pads,
            dilations,
            output_channels,
            group
        );

        builder.add_computational_memlet(block, input_access, conv_node, "X", {}, tensor_type, DebugInfo());
        builder.add_computational_memlet(block, weights_access, conv_node, "W", {}, weights_type, DebugInfo());
        builder.add_computational_memlet(block, conv_node, "Y", output_access, {}, tensor_type, DebugInfo());

        return block;
    }
};

} // namespace

//
// ==================== POSITIVE TESTS ====================
//

TEST(LocalBufferReuseTest, ConvBatchNorm_Elimination_Applied) {
    ConvBatchNormSetup setup;
    setup.setup_containers();

    // Block 1: malloc -> ptr1
    setup.add_malloc_block("ptr1", setup.malloc_size);
    // Block 2: ptr1 -> ref1 (reference)
    setup.add_reference_block("ptr1", "ref1");
    // Block 3: Conv -> ref1
    setup.add_conv_block("ref1");
    // Block 4: malloc -> ptr2
    setup.add_malloc_block("ptr2", setup.malloc_size);
    // Block 5: ptr2 -> ref2 (reference)
    setup.add_reference_block("ptr2", "ref2");
    // Block 6: BatchNorm: ref1 -> ref2
    setup.add_batchnorm_block("ref1", "ref2");
    // Block 7: free ptr1
    setup.add_free_block("ptr1");
    // Block 8: free ptr2
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();
    EXPECT_EQ(original_size, 8);

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConvBatchNormEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_TRUE(applied);

    sdfg = builder_opt.move();
    // Block 4 (second malloc) and one free block should be removed
    EXPECT_LT(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, BatchNormReLU_Elimination_Applied) {
    BatchNormReLUSetup setup;
    setup.setup_containers();

    // Block 1: malloc -> ptr1
    setup.add_malloc_block("ptr1", setup.malloc_size);
    // Block 2: ptr1 -> ref1 (reference)
    setup.add_reference_block("ptr1", "ref1");
    // Block 3: BatchNorm -> ref1
    setup.add_batchnorm_block("ref1");
    // Block 4: malloc -> ptr2
    setup.add_malloc_block("ptr2", setup.malloc_size);
    // Block 5: ptr2 -> ref2 (reference)
    setup.add_reference_block("ptr2", "ref2");
    // Block 6: ReLU: ref1 -> ref2
    setup.add_relu_block("ref1", "ref2");
    // Block 7: free ptr1
    setup.add_free_block("ptr1");
    // Block 8: free ptr2
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();
    EXPECT_EQ(original_size, 8);

    // Apply pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BatchNormReLUEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_TRUE(applied);

    sdfg = builder_opt.move();
    EXPECT_LT(sdfg->root().size(), original_size);
}

//
// ==================== NEGATIVE TESTS ====================
//

TEST(LocalBufferReuseTest, Negative_DifferentMallocSizes) {
    BatchNormReLUSetup setup;
    setup.setup_containers();

    // Block 1: malloc -> ptr1 with size X
    setup.add_malloc_block("ptr1", setup.malloc_size);
    // Block 2: ptr1 -> ref1 (reference)
    setup.add_reference_block("ptr1", "ref1");
    // Block 3: BatchNorm -> ref1
    setup.add_batchnorm_block("ref1");
    // Block 4: malloc -> ptr2 with DIFFERENT size (size*2)
    auto different_size = symbolic::integer(1 * 64 * 32 * 32 * 4 * 2); // Different size
    setup.add_malloc_block("ptr2", different_size);
    // Block 5: ptr2 -> ref2 (reference)
    setup.add_reference_block("ptr2", "ref2");
    // Block 6: ReLU: ref1 -> ref2
    setup.add_relu_block("ref1", "ref2");
    // Block 7: free ptr1
    setup.add_free_block("ptr1");
    // Block 8: free ptr2
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply pass - should NOT be applied due to size mismatch
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BatchNormReLUEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, Negative_WrongLibraryNodeType) {
    // Use ConvBatchNormElimination pass but provide BatchNorm -> ReLU pattern
    BatchNormReLUSetup setup;
    setup.setup_containers();

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_batchnorm_block("ref1");
    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_relu_block("ref1", "ref2"); // ReLU instead of BatchNorm
    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply ConvBatchNorm pass to BatchNorm->ReLU pattern - should NOT match (wrong T type)
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConvBatchNormEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, Negative_MissingReferenceBlock) {
    BatchNormReLUSetup setup;
    setup.setup_containers();

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_batchnorm_block("ref1");
    setup.add_malloc_block("ptr2", setup.malloc_size);
    // Missing: reference block for ptr2 -> ref2
    // Add a dummy block instead
    {
        auto& root = setup.builder.subject().root();
        auto& block = setup.builder.add_block(root);
        // Empty block or block with something other than reference
        auto& access = setup.builder.add_access(block, "ptr2");
        auto& access2 = setup.builder.add_access(block, "ref2");
        auto& tasklet = setup.builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        setup.builder.add_computational_memlet(block, access, tasklet, "_in", {}, setup.ptr_desc, DebugInfo());
        setup.builder.add_computational_memlet(block, tasklet, "_out", access2, {}, setup.ptr_desc, DebugInfo());
    }
    setup.add_relu_block("ref1", "ref2");
    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply pass - should NOT be applied due to missing reference pattern
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BatchNormReLUEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, Negative_TooFewBlocks) {
    // Create a pattern with fewer than 6 blocks
    BatchNormReLUSetup setup;
    setup.setup_containers();

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_batchnorm_block("ref1");
    // Only 3 blocks - not enough for pattern match

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();
    EXPECT_EQ(original_size, 3);

    // Apply pass - should NOT be applied due to insufficient blocks
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BatchNormReLUEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, Negative_WrongOutputConnection) {
    BatchNormReLUSetup setup;
    setup.setup_containers();

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_batchnorm_block("ref1");
    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");

    // Create ReLU that outputs to wrong container (ref1 instead of ref2)
    {
        auto& root = setup.builder.subject().root();
        auto& block = setup.builder.add_block(root);

        auto& input_access = setup.builder.add_access(block, "ref1");
        auto& output_access = setup.builder.add_access(block, "ref1"); // Wrong: should be ref2

        auto& relu_node = setup.builder.add_library_node<math::tensor::ReLUNode>(
            block, DebugInfo(), std::vector<symbolic::Expression>(setup.tensor_shape.begin(), setup.tensor_shape.end())
        );

        setup.builder.add_computational_memlet(block, input_access, relu_node, "X", {}, setup.tensor_type, DebugInfo());
        setup.builder.add_computational_memlet(block, relu_node, "Y", output_access, {}, setup.tensor_type, DebugInfo());
    }

    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply pass - should NOT be applied due to wrong output connection
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BatchNormReLUEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, Negative_ExtraUsersOfMallocContainer) {
    BatchNormReLUSetup setup;
    setup.setup_containers();

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_batchnorm_block("ref1");
    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_relu_block("ref1", "ref2");

    // Add extra usage of ptr2 before free (extra user)
    {
        auto& root = setup.builder.subject().root();
        auto& block = setup.builder.add_block(root);
        auto& access = setup.builder.add_access(block, "ptr2");
        setup.builder.add_container("extra_out", setup.ptr_desc);
        auto& out_access = setup.builder.add_access(block, "extra_out");
        auto& tasklet = setup.builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        setup.builder.add_computational_memlet(block, access, tasklet, "_in", {}, setup.ptr_desc, DebugInfo());
        setup.builder.add_computational_memlet(block, tasklet, "_out", out_access, {}, setup.ptr_desc, DebugInfo());
    }

    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply pass - should NOT be applied due to extra users
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BatchNormReLUEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, Negative_NoFreeBlock) {
    BatchNormReLUSetup setup;
    setup.setup_containers();

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_batchnorm_block("ref1");
    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_relu_block("ref1", "ref2");
    setup.add_free_block("ptr1");
    // Missing: free block for ptr2

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply pass - should NOT be applied due to missing free block
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BatchNormReLUEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, Negative_NonEmptyTransition) {
    BatchNormReLUSetup setup;
    setup.setup_containers();

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_batchnorm_block("ref1");

    // Add malloc block with non-empty transition (assignments)
    {
        auto& root = setup.builder.subject().root();
        setup.builder.add_container("extra_sym", types::Scalar(types::PrimitiveType::Int32));
        auto& block =
            setup.builder.add_block(root, {{symbolic::symbol("extra_sym"), symbolic::integer(42)}}); // Non-empty
                                                                                                     // transition
        auto& access = setup.builder.add_access(block, "ptr2");
        auto& malloc_node = setup.builder.add_library_node<stdlib::MallocNode>(block, DebugInfo(), setup.malloc_size);
        setup.builder.add_computational_memlet(block, malloc_node, "_ret", access, {}, setup.ptr_desc, DebugInfo());
    }

    setup.add_reference_block("ptr2", "ref2");
    setup.add_relu_block("ref1", "ref2");
    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply pass - should NOT be applied due to non-empty transition
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BatchNormReLUEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, Negative_Block1NotMalloc) {
    BatchNormReLUSetup setup;
    setup.setup_containers();

    // Block 1: NOT a malloc - just a tasklet
    {
        auto& root = setup.builder.subject().root();
        auto& block = setup.builder.add_block(root);
        auto& access = setup.builder.add_access(block, "ptr1");
        auto& constant = setup.builder.add_constant(block, "0", setup.element_desc);
        auto& tasklet = setup.builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        setup.builder.add_computational_memlet(block, constant, tasklet, "_in", {}, setup.element_desc, DebugInfo());
        setup.builder.add_computational_memlet(block, tasklet, "_out", access, {}, setup.ptr_desc, DebugInfo());
    }

    setup.add_reference_block("ptr1", "ref1");
    setup.add_batchnorm_block("ref1");
    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_relu_block("ref1", "ref2");
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply pass - should NOT be applied because block 1 is not malloc
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::BatchNormReLUEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, Pipeline_AllPasses) {
    BatchNormReLUSetup setup;
    setup.setup_containers();

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_batchnorm_block("ref1");
    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_relu_block("ref1", "ref2");
    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply full pipeline
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto pipeline = passes::local_buffer_reuse_pipeline();
    pipeline.run(builder_opt, analysis_manager);

    sdfg = builder_opt.move();
    EXPECT_LT(sdfg->root().size(), original_size);
}

//
// ==================== 5-LIB-NODE TESTS ====================
//

TEST(LocalBufferReuseTest, FiveNodeChain_Elimination_Applied) {
    ConvBatchNormReLUConvBatchNormSetup setup;
    setup.setup_containers();

    // Pattern: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
    // Block structure (5 * 3 = 15 blocks + 5 free blocks = 20 total):
    // 1. malloc -> ptr1
    // 2. ptr1 -> ref1
    // 3. Conv: input -> ref1
    // 4. malloc -> ptr2
    // 5. ptr2 -> ref2
    // 6. BatchNorm: ref1 -> ref2
    // 7. malloc -> ptr3
    // 8. ptr3 -> ref3
    // 9. ReLU: ref2 -> ref3
    // 10. malloc -> ptr4
    // 11. ptr4 -> ref4
    // 12. Conv: ref3 -> ref4
    // 13. malloc -> ptr5
    // 14. ptr5 -> ref5
    // 15. BatchNorm: ref4 -> ref5
    // 16-20. free blocks

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_first_conv_block("ref1");

    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_batchnorm_block("ref1", "ref2");

    setup.add_malloc_block("ptr3", setup.malloc_size);
    setup.add_reference_block("ptr3", "ref3");
    setup.add_relu_block("ref2", "ref3");

    setup.add_malloc_block("ptr4", setup.malloc_size);
    setup.add_reference_block("ptr4", "ref4");
    setup.add_conv_block("ref3", "ref4");

    setup.add_malloc_block("ptr5", setup.malloc_size);
    setup.add_reference_block("ptr5", "ref5");
    setup.add_batchnorm_block("ref4", "ref5");

    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");
    setup.add_free_block("ptr3");
    setup.add_free_block("ptr4");
    setup.add_free_block("ptr5");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();
    EXPECT_EQ(original_size, 20); // 15 pattern blocks + 5 free blocks

    // Apply 5-node pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConvBatchNormReLUConvBatchNormEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_TRUE(applied);

    sdfg = builder_opt.move();
    // 4 malloc blocks and 4 free blocks should be removed (keeping first malloc/ref/free)
    // Expected: 20 - 4 (malloc blocks 2-5) - 4 (free blocks 2-5) = 12
    EXPECT_EQ(sdfg->root().size(), 12);
}

TEST(LocalBufferReuseTest, FiveNodeChain_DifferentSizes_NotApplied) {
    ConvBatchNormReLUConvBatchNormSetup setup;
    setup.setup_containers();

    // Same pattern but with different malloc sizes - should NOT be applied
    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_first_conv_block("ref1");

    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_batchnorm_block("ref1", "ref2");

    // Different malloc size for ptr3
    setup.add_malloc_block("ptr3", symbolic::integer(999));
    setup.add_reference_block("ptr3", "ref3");
    setup.add_relu_block("ref2", "ref3");

    setup.add_malloc_block("ptr4", setup.malloc_size);
    setup.add_reference_block("ptr4", "ref4");
    setup.add_conv_block("ref3", "ref4");

    setup.add_malloc_block("ptr5", setup.malloc_size);
    setup.add_reference_block("ptr5", "ref5");
    setup.add_batchnorm_block("ref4", "ref5");

    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");
    setup.add_free_block("ptr3");
    setup.add_free_block("ptr4");
    setup.add_free_block("ptr5");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply 5-node pass - should NOT be applied
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConvBatchNormReLUConvBatchNormEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, FiveNodeChain_Pipeline_LongestFirst) {
    ConvBatchNormReLUConvBatchNormSetup setup;
    setup.setup_containers();

    // Create a 5-node pattern - pipeline should match this with 5-node pass first
    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_first_conv_block("ref1");

    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_batchnorm_block("ref1", "ref2");

    setup.add_malloc_block("ptr3", setup.malloc_size);
    setup.add_reference_block("ptr3", "ref3");
    setup.add_relu_block("ref2", "ref3");

    setup.add_malloc_block("ptr4", setup.malloc_size);
    setup.add_reference_block("ptr4", "ref4");
    setup.add_conv_block("ref3", "ref4");

    setup.add_malloc_block("ptr5", setup.malloc_size);
    setup.add_reference_block("ptr5", "ref5");
    setup.add_batchnorm_block("ref4", "ref5");

    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");
    setup.add_free_block("ptr3");
    setup.add_free_block("ptr4");
    setup.add_free_block("ptr5");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();
    EXPECT_EQ(original_size, 20);

    // Apply full pipeline - should use 5-node pass
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    auto pipeline = passes::local_buffer_reuse_pipeline();
    pipeline.run(builder_opt, analysis_manager);

    sdfg = builder_opt.move();
    // Should eliminate 4 mallocs and 4 frees = 8 blocks
    EXPECT_EQ(sdfg->root().size(), 12);
}

TEST(LocalBufferReuseTest, FiveNodeChain_MissingFreeBlock_NotApplied) {
    ConvBatchNormReLUConvBatchNormSetup setup;
    setup.setup_containers();

    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_first_conv_block("ref1");

    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_batchnorm_block("ref1", "ref2");

    setup.add_malloc_block("ptr3", setup.malloc_size);
    setup.add_reference_block("ptr3", "ref3");
    setup.add_relu_block("ref2", "ref3");

    setup.add_malloc_block("ptr4", setup.malloc_size);
    setup.add_reference_block("ptr4", "ref4");
    setup.add_conv_block("ref3", "ref4");

    setup.add_malloc_block("ptr5", setup.malloc_size);
    setup.add_reference_block("ptr5", "ref5");
    setup.add_batchnorm_block("ref4", "ref5");

    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");
    // Missing: free block for ptr3
    setup.add_free_block("ptr4");
    setup.add_free_block("ptr5");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply 5-node pass - should NOT be applied due to missing free block
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConvBatchNormReLUConvBatchNormEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}

TEST(LocalBufferReuseTest, FiveNodeChain_TooFewBlocks_NotApplied) {
    ConvBatchNormReLUConvBatchNormSetup setup;
    setup.setup_containers();

    // Only create 4 triples instead of 5 - should NOT match 5-node pattern
    setup.add_malloc_block("ptr1", setup.malloc_size);
    setup.add_reference_block("ptr1", "ref1");
    setup.add_first_conv_block("ref1");

    setup.add_malloc_block("ptr2", setup.malloc_size);
    setup.add_reference_block("ptr2", "ref2");
    setup.add_batchnorm_block("ref1", "ref2");

    setup.add_malloc_block("ptr3", setup.malloc_size);
    setup.add_reference_block("ptr3", "ref3");
    setup.add_relu_block("ref2", "ref3");

    setup.add_malloc_block("ptr4", setup.malloc_size);
    setup.add_reference_block("ptr4", "ref4");
    setup.add_conv_block("ref3", "ref4");
    // Missing: 5th triple

    setup.add_free_block("ptr1");
    setup.add_free_block("ptr2");
    setup.add_free_block("ptr3");
    setup.add_free_block("ptr4");

    auto sdfg = setup.builder.move();
    size_t original_size = sdfg->root().size();

    // Apply 5-node pass - should NOT be applied (not enough blocks)
    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::ConvBatchNormReLUConvBatchNormEliminationPass pass;
    bool applied = pass.run(builder_opt, analysis_manager);

    EXPECT_FALSE(applied);

    sdfg = builder_opt.move();
    EXPECT_EQ(sdfg->root().size(), original_size);
}
