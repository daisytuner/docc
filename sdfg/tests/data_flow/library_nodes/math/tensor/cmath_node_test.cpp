#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/cmath_node.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/memlet.h"
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

void TestCMathNode(math::cmath::CMathFunction function, std::vector<size_t> shape_dims) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::PrimitiveType out_desc_primitive;
    if (function == math::cmath::CMathFunction::lrint || function == math::cmath::CMathFunction::llrint ||
        function == math::cmath::CMathFunction::lround || function == math::cmath::CMathFunction::llround) {
        out_desc_primitive = types::PrimitiveType::Int64;
    } else {
        out_desc_primitive = types::PrimitiveType::Float;
    }
    types::Scalar out_desc(out_desc_primitive);
    types::Pointer out_desc_ptr(out_desc);
    types::Scalar desc(types::PrimitiveType::Float);
    types::Pointer desc_ptr(desc);

    size_t function_arity = math::cmath::cmath_function_to_arity(function);
    ASSERT_GE(function_arity, 1);
    ASSERT_LE(function_arity, 3);

    std::vector<std::string> outputs, inputs;
    inputs.reserve(function_arity);
    switch (function_arity) {
        case 3:
            builder.add_container("d", desc_ptr, true);
            inputs.push_back("_in3");
        case 2:
            builder.add_container("c", desc_ptr, true);
            inputs.push_back("_in2");
        case 1:
            builder.add_container("b", desc_ptr, true);
            builder.add_container("a", out_desc_ptr, true);
            inputs.push_back("_in1");
            outputs.push_back("_out");
            break;
    }

    auto& block = builder.add_block(root);

    std::vector<symbolic::Expression> shape;
    for (auto d : shape_dims) {
        shape.push_back(symbolic::integer(d));
    }
    types::Tensor tensor_type(types::PrimitiveType::Float, shape);
    types::Tensor out_tensor_type(out_desc_primitive, shape);

    auto& tensor_node = static_cast<math::tensor::CMathTensorNode&>(
        builder.add_library_node<math::tensor::CMathTensorNode>(block, DebugInfo(), function, outputs, inputs, shape)
    );

    switch (function_arity) {
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
            builder.add_computational_memlet(block, tensor_node, "_out", a_node, {}, out_tensor_type);
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
    EXPECT_EQ(dfg.nodes().size(), function_arity + 2);
    EXPECT_EQ(dfg.tasklets().size(), 0);
    EXPECT_EQ(dfg.library_nodes().size(), 1);
    EXPECT_EQ(dfg.data_nodes().size(), function_arity + 1);

    auto* cmath_node = dynamic_cast<math::cmath::CMathNode*>(*dfg.library_nodes().begin());
    ASSERT_TRUE(cmath_node);
    EXPECT_EQ(cmath_node->function(), function);

    for (auto& edge : dfg.edges()) {
        if (edge.subset().size() > 0) {
            ASSERT_EQ(edge.subset().size(), loop_indvars.size());
            for (size_t i = 0; i < loop_indvars.size(); i++) {
                EXPECT_TRUE(symbolic::null_safe_eq(edge.subset().at(i), loop_indvars.at(i)));
            }
        }
    }
}

#define REGISTER_TEST_WITH_DIM(Function, Dim)                      \
    TEST(CMathTensorNodeTest, Function##_##Dim##D) {               \
        std::vector<size_t> dims;                                  \
        for (size_t i = 0; i < Dim; ++i) {                         \
            dims.push_back(32);                                    \
        }                                                          \
        TestCMathNode(math::cmath::CMathFunction::Function, dims); \
    }

#define REGISTER_TEST(Function)         \
    REGISTER_TEST_WITH_DIM(Function, 1) \
    REGISTER_TEST_WITH_DIM(Function, 2) \
    REGISTER_TEST_WITH_DIM(Function, 3) \
    REGISTER_TEST_WITH_DIM(Function, 4)

REGISTER_TEST(sin);
REGISTER_TEST(cos);
REGISTER_TEST(tan);
REGISTER_TEST(asin);
REGISTER_TEST(acos);
REGISTER_TEST(atan);
REGISTER_TEST(atan2);
REGISTER_TEST(sinh);
REGISTER_TEST(cosh);
REGISTER_TEST(tanh);
REGISTER_TEST(asinh);
REGISTER_TEST(acosh);
REGISTER_TEST(atanh);
REGISTER_TEST(exp);
REGISTER_TEST(exp2);
REGISTER_TEST(exp10);
REGISTER_TEST(expm1);
REGISTER_TEST(log);
REGISTER_TEST(log10);
REGISTER_TEST(log2);
REGISTER_TEST(log1p);
REGISTER_TEST(pow);
REGISTER_TEST(sqrt);
REGISTER_TEST(cbrt);
REGISTER_TEST(hypot);
REGISTER_TEST(erf);
REGISTER_TEST(erfc);
REGISTER_TEST(tgamma);
REGISTER_TEST(lgamma);
REGISTER_TEST(fabs);
REGISTER_TEST(ceil);
REGISTER_TEST(floor);
REGISTER_TEST(trunc);
REGISTER_TEST(round);
REGISTER_TEST(lround);
REGISTER_TEST(llround);
REGISTER_TEST(roundeven);
REGISTER_TEST(nearbyint);
REGISTER_TEST(rint);
REGISTER_TEST(lrint);
REGISTER_TEST(llrint);
REGISTER_TEST(fmod);
REGISTER_TEST(remainder);
REGISTER_TEST(frexp);
REGISTER_TEST(ldexp);
REGISTER_TEST(modf);
REGISTER_TEST(scalbn);
REGISTER_TEST(scalbln);
REGISTER_TEST(ilogb);
REGISTER_TEST(logb);
REGISTER_TEST(nextafter);
REGISTER_TEST(nexttoward);
REGISTER_TEST(copysign);
REGISTER_TEST(fmax);
REGISTER_TEST(fmin);
REGISTER_TEST(fdim);
REGISTER_TEST(fma);
