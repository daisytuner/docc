#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(BatchNormTest, BatchNorm_2D_Simple) {
    // Test simple 2D batch normalization with shapes: Batch[B], Var[B], E[B], Gamma[B], Beta[B], B_out[B]
    builder::StructuredSDFGBuilder builder("sdfg_batchnorm_2d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    builder.add_container("Batch", desc_ptr, true);
    builder.add_container("Var", desc_ptr, true);
    builder.add_container("E", desc_ptr, true);
    builder.add_container("Gamma", desc_ptr, true);
    builder.add_container("Beta", desc_ptr, true);
    builder.add_container("B_out", desc_ptr, true);

    auto& block = builder.add_block(sdfg.root());

    auto& batch_node = builder.add_access(block, "Batch");
    auto& var_node = builder.add_access(block, "Var");
    auto& e_node = builder.add_access(block, "E");
    auto& gamma_node = builder.add_access(block, "Gamma");
    auto& beta_node = builder.add_access(block, "Beta");
    auto& b_out_node = builder.add_access(block, "B_out");
    auto& epsilon_node = builder.add_constant(block, "0.00001", desc);

    symbolic::MultiExpression batch_shape = {
        symbolic::integer(1), symbolic::integer(10), symbolic::integer(8), symbolic::integer(8)
    };
    symbolic::MultiExpression norm_shape = {symbolic::integer(10)};

    types::Tensor batch_tensor(desc.primitive_type(), batch_shape);
    types::Tensor var_tensor(desc.primitive_type(), norm_shape);
    types::Tensor e_tensor(desc.primitive_type(), norm_shape);
    types::Tensor gamma_tensor(desc.primitive_type(), norm_shape);
    types::Tensor beta_tensor(desc.primitive_type(), norm_shape);
    types::Tensor b_out_tensor(desc.primitive_type(), batch_shape);

    auto& batchnorm_node =
        dynamic_cast<math::tensor::BatchNormNode&>(builder.add_library_node<math::tensor::BatchNormNode>(
            block, DebugInfo(), math::tensor::TensorLayout(batch_shape), types::Float
        ));

    builder.add_computational_memlet(block, batch_node, batchnorm_node, "Batch", {}, batch_tensor, block.debug_info());
    builder.add_computational_memlet(block, var_node, batchnorm_node, "Var", {}, var_tensor, block.debug_info());
    builder.add_computational_memlet(block, e_node, batchnorm_node, "E", {}, e_tensor, block.debug_info());
    builder.add_computational_memlet(block, gamma_node, batchnorm_node, "Gamma", {}, gamma_tensor, block.debug_info());
    builder.add_computational_memlet(block, beta_node, batchnorm_node, "Beta", {}, beta_tensor, block.debug_info());
    builder.add_computational_memlet(block, epsilon_node, batchnorm_node, "epsilon", {}, desc, block.debug_info());
    builder.add_computational_memlet(block, b_out_node, batchnorm_node, "B_out", {}, b_out_tensor, block.debug_info());

    builder.subject().validate();

    dump_sdfg(builder.subject(), "0.pre-expand");

    analysis::AnalysisManager ana(builder.subject());

    batchnorm_node.expand(builder, ana);

    dump_sdfg(builder.subject(), "1.post-expand");

    builder.subject().validate();

    EXPECT_EQ(builder.subject().root().size(), 1);
}
