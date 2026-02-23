#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/tasklet_node.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

using namespace sdfg;

void TestTaskletNode(data_flow::TaskletCode code, std::vector<size_t> shape_dims) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::PrimitiveType desc_primitive;
    if (data_flow::is_integer(code)) {
        desc_primitive = types::PrimitiveType::Int32;
    } else {
        desc_primitive = types::PrimitiveType::Float;
    }
    types::Scalar desc(desc_primitive);
    types::Pointer desc_ptr(desc);

    size_t code_arity = data_flow::arity(code);
    ASSERT_GE(code_arity, 1);
    ASSERT_LE(code_arity, 3);

    std::vector<std::string> outputs, inputs;
    inputs.reserve(code_arity);
    switch (code_arity) {
        case 3:
            builder.add_container("d", desc_ptr, true);
            inputs.push_back("_in3");
        case 2:
            builder.add_container("c", desc_ptr, true);
            inputs.push_back("_in2");
        case 1:
            builder.add_container("b", desc_ptr, true);
            builder.add_container("a", desc_ptr, true);
            inputs.push_back("_in1");
            outputs.push_back("_out");
            break;
    }

    auto& block = builder.add_block(root);

    std::vector<symbolic::Expression> shape;
    for (auto d : shape_dims) {
        shape.push_back(symbolic::integer(d));
    }
    types::Tensor tensor_type(desc_primitive, shape);

    auto& tensor_node = static_cast<math::tensor::TaskletTensorNode&>(
        builder.add_library_node<math::tensor::TaskletTensorNode>(block, DebugInfo(), code, outputs, inputs, shape)
    );

    switch (code_arity) {
        case 3: {
            auto& d_node = builder.add_access(block, "d");
            builder.add_computational_memlet(block, d_node, tensor_node, "_in3", {}, tensor_type);
        }
        case 2: {
            auto& c_node = builder.add_access(block, "c");
            builder.add_computational_memlet(block, c_node, tensor_node, "_in2", {}, tensor_type);
        }
        case 1: {
            auto& b_node = builder.add_access(block, "b");
            builder.add_computational_memlet(block, b_node, tensor_node, "_in1", {}, tensor_type);
            auto& a_node = builder.add_access(block, "a");
            builder.add_computational_memlet(block, tensor_node, "_out", a_node, {}, tensor_type);
        }
    }

    EXPECT_NO_THROW(sdfg.validate());

    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(tensor_node.expand(builder, analysis_manager));

    EXPECT_NO_THROW(sdfg.validate());

    EXPECT_EQ(root.size(), 1);
    ASSERT_GE(root.size(), 1);
    auto* new_sequence = dynamic_cast<structured_control_flow::Sequence*>(&root.at(0).first);
    ASSERT_TRUE(new_sequence);

    data_flow::Subset loop_indvars;
    structured_control_flow::Sequence* current_scope = new_sequence;
    for (size_t i = 0; i < shape_dims.size(); ++i) {
        EXPECT_EQ(current_scope->size(), 1);
        ASSERT_GE(current_scope->size(), 1);
        auto* map_loop = dynamic_cast<structured_control_flow::Map*>(&current_scope->at(0).first);
        ASSERT_TRUE(map_loop);
        auto indvar = map_loop->indvar();
        EXPECT_TRUE(symbolic::null_safe_eq(map_loop->init(), symbolic::zero()));
        EXPECT_TRUE(symbolic::null_safe_eq(map_loop->condition(), symbolic::Lt(indvar, shape[i])));
        EXPECT_TRUE(symbolic::null_safe_eq(map_loop->update(), symbolic::add(indvar, symbolic::one())));
        loop_indvars.push_back(indvar);
        current_scope = &map_loop->root();
    }

    EXPECT_EQ(current_scope->size(), 1);
    ASSERT_GE(current_scope->size(), 1);
    auto* code_block = dynamic_cast<structured_control_flow::Block*>(&current_scope->at(0).first);
    ASSERT_TRUE(code_block);

    auto& dfg = code_block->dataflow();
    EXPECT_EQ(dfg.nodes().size(), code_arity + 2);
    EXPECT_EQ(dfg.tasklets().size(), 1);
    EXPECT_EQ(dfg.library_nodes().size(), 0);
    EXPECT_EQ(dfg.data_nodes().size(), code_arity + 1);

    auto* tasklet = *dfg.tasklets().begin();
    ASSERT_TRUE(tasklet);
    EXPECT_EQ(tasklet->code(), code);

    for (auto& edge : dfg.edges()) {
        if (edge.subset().size() > 0) {
            ASSERT_EQ(edge.subset().size(), loop_indvars.size());
            for (size_t i = 0; i < loop_indvars.size(); i++) {
                EXPECT_TRUE(symbolic::null_safe_eq(edge.subset().at(i), loop_indvars.at(i)));
            }
        }
    }
}

#define REGISTER_TEST_WITH_DIM(Code, Dim)                    \
    TEST(TaskletTensorNodeTest, Code##_##Dim##D) {           \
        std::vector<size_t> dims;                            \
        for (size_t i = 0; i < Dim; ++i) {                   \
            dims.push_back(32);                              \
        }                                                    \
        TestTaskletNode(data_flow::TaskletCode::Code, dims); \
    }

#define REGISTER_TEST(Code)         \
    REGISTER_TEST_WITH_DIM(Code, 1) \
    REGISTER_TEST_WITH_DIM(Code, 2) \
    REGISTER_TEST_WITH_DIM(Code, 3) \
    REGISTER_TEST_WITH_DIM(Code, 4)

REGISTER_TEST(assign)
REGISTER_TEST(fp_neg)
REGISTER_TEST(fp_add)
REGISTER_TEST(fp_sub)
REGISTER_TEST(fp_mul)
REGISTER_TEST(fp_div)
REGISTER_TEST(fp_rem)
REGISTER_TEST(fp_fma)
REGISTER_TEST(fp_oeq)
REGISTER_TEST(fp_one)
REGISTER_TEST(fp_oge)
REGISTER_TEST(fp_ogt)
REGISTER_TEST(fp_ole)
REGISTER_TEST(fp_olt)
REGISTER_TEST(fp_ord)
REGISTER_TEST(fp_ueq)
REGISTER_TEST(fp_une)
REGISTER_TEST(fp_ugt)
REGISTER_TEST(fp_uge)
REGISTER_TEST(fp_ult)
REGISTER_TEST(fp_ule)
REGISTER_TEST(fp_uno)
REGISTER_TEST(int_add)
REGISTER_TEST(int_sub)
REGISTER_TEST(int_mul)
REGISTER_TEST(int_sdiv)
REGISTER_TEST(int_srem)
REGISTER_TEST(int_udiv)
REGISTER_TEST(int_urem)
REGISTER_TEST(int_and)
REGISTER_TEST(int_or)
REGISTER_TEST(int_xor)
REGISTER_TEST(int_shl)
REGISTER_TEST(int_ashr)
REGISTER_TEST(int_lshr)
REGISTER_TEST(int_smin)
REGISTER_TEST(int_smax)
REGISTER_TEST(int_scmp)
REGISTER_TEST(int_umin)
REGISTER_TEST(int_umax)
REGISTER_TEST(int_ucmp)
REGISTER_TEST(int_eq)
REGISTER_TEST(int_ne)
REGISTER_TEST(int_sge)
REGISTER_TEST(int_sgt)
REGISTER_TEST(int_sle)
REGISTER_TEST(int_slt)
REGISTER_TEST(int_uge)
REGISTER_TEST(int_ugt)
REGISTER_TEST(int_ule)
REGISTER_TEST(int_ult)
REGISTER_TEST(int_abs)
