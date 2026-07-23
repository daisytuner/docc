#include "sdfg/data_flow/library_nodes/math/tensor/copy_node.h"

#include <gtest/gtest.h>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

struct TensorCopyNodeTestReturn {
    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> dims;
    data_flow::Subset subset_x;
    data_flow::Subset subset_y;
};

void create_and_test(
    math::tensor::TensorLayout& layout_x, math::tensor::TensorLayout& layout_y, TensorCopyNodeTestReturn& test_return
) {
    test_return.dims.clear();
    test_return.subset_x.clear();
    test_return.subset_y.clear();

    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Float);
    types::Pointer desc_type(base_type);
    if (layout_x.is_scalar()) {
        builder.add_container("X", base_type, true);
    } else {
        builder.add_container("X", desc_type, true);
    }
    if (layout_y.is_scalar()) {
        builder.add_container("Y", base_type, true);
    } else {
        builder.add_container("Y", desc_type, true);
    }

    types::Tensor X_tensor(base_type, layout_x);
    types::Tensor Y_tensor(base_type, layout_y);

    auto& block = builder.add_block(root);
    auto& X_access = builder.add_access(block, "X");
    auto& Y_access = builder.add_access(block, "Y");
    auto& libnode =
        builder.add_library_node<math::tensor::TensorCopyNode>(block, sdfg::DebugInfo(), layout_x, layout_y);
    builder.add_computational_memlet(block, X_access, libnode, "X", {}, X_tensor);
    builder.add_computational_memlet(block, Y_access, libnode, "Y", {}, Y_tensor);

    ASSERT_NO_THROW(sdfg.validate());
    dump_sdfg(sdfg, "0.before");

    analysis::AnalysisManager analysis_manager(sdfg);
    passes::LibraryNodeExpansionPass expansion;
    ASSERT_TRUE(expansion.run(builder, analysis_manager));
    dump_sdfg(sdfg, "1.after");

    ASSERT_EQ(root.size(), 1);
    auto* current_seq = sdfg::dyn_cast<structured_control_flow::Sequence*>(&root.at(0).first);
    ASSERT_NE(current_seq, nullptr);
    ASSERT_EQ(current_seq->size(), 1);

    while (structured_control_flow::Map::classof(current_seq->at(0).first)) {
        auto* map = sdfg::dyn_cast<structured_control_flow::Map*>(&current_seq->at(0).first);
        ASSERT_TRUE(symbolic::eq(map->init(), symbolic::zero()));
        auto indvar = map->indvar();
        ASSERT_TRUE(symbolic::eq(map->update(), symbolic::add(indvar, symbolic::one())));
        auto num_iterations = map->num_iterations();
        ASSERT_FALSE(num_iterations.is_null());
        test_return.dims.push_back({indvar, num_iterations});

        current_seq = &map->root();
        ASSERT_EQ(current_seq->size(), 1);
    }

    auto* comp_block = sdfg::dyn_cast<structured_control_flow::Block*>(&current_seq->at(0).first);
    ASSERT_NE(comp_block, nullptr);

    auto& dfg = comp_block->dataflow();
    ASSERT_EQ(dfg.tasklets().size(), 1);
    ASSERT_EQ(dfg.library_nodes().size(), 0);
    ASSERT_EQ(dfg.data_nodes().size(), 2);

    auto* tasklet = *dfg.tasklets().begin();
    ASSERT_NE(tasklet, nullptr);
    ASSERT_EQ(tasklet->inputs().size(), 1);

    auto* iedge = dfg.in_edge_for_connector(*tasklet, tasklet->input(0));
    ASSERT_NE(iedge, nullptr);

    auto* access_X = sdfg::dyn_cast<data_flow::AccessNode*>(&iedge->src());
    ASSERT_NE(access_X, nullptr);
    ASSERT_EQ(access_X->data(), "X");

    auto oedges = dfg.out_edges_for_connector(*tasklet, tasklet->output());
    ASSERT_EQ(oedges.size(), 1);
    auto* oedge = oedges[0];
    ASSERT_NE(oedge, nullptr);

    auto* access_Y = sdfg::dyn_cast<data_flow::AccessNode*>(&oedge->dst());
    ASSERT_NE(access_Y, nullptr);
    ASSERT_EQ(access_Y->data(), "Y");

    test_return.subset_x = iedge->subset();
    test_return.subset_y = oedge->subset();
}

bool subsets_eq(const data_flow::Subset& subset1, const data_flow::Subset& subset2) {
    int size = subset1.size();
    if (size != subset2.size()) {
        return false;
    }
    for (int i = 0; i < size; i++) {
        if (!symbolic::eq(subset1[i], subset2[i])) {
            return false;
        }
    }
    return true;
}

TEST(TensorCopyNodeTest, identity) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(3);
    math::tensor::TensorLayout layout_x({a, b});
    math::tensor::TensorLayout layout_y({a, b});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 2);
    auto i = test_return.dims[0].first;
    auto j = test_return.dims[1].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(symbolic::eq(test_return.dims[1].second, b));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {i, j}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i, j}));
}

TEST(TensorCopyNodeTest, identity_empty) {
    math::tensor::TensorLayout layout_x({});
    math::tensor::TensorLayout layout_y({});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 0);
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {}));
}

TEST(TensorCopyNodeTest, permutation_2d) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(3);
    math::tensor::TensorLayout layout_x({a, b});
    math::tensor::TensorLayout layout_y({b, a});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 2);
    auto i = test_return.dims[0].first;
    auto j = test_return.dims[1].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, b));
    ASSERT_TRUE(symbolic::eq(test_return.dims[1].second, a));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {j, i}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i, j}));
}

TEST(TensorCopyNodeTest, permutation_3d) {
    auto a = symbolic::integer(1);
    auto b = symbolic::integer(2);
    auto c = symbolic::integer(3);
    math::tensor::TensorLayout layout_x({a, b, c});
    math::tensor::TensorLayout layout_y({c, b, a});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 3);
    auto i = test_return.dims[0].first;
    auto j = test_return.dims[1].first;
    auto k = test_return.dims[2].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, c));
    ASSERT_TRUE(symbolic::eq(test_return.dims[1].second, b));
    ASSERT_TRUE(symbolic::eq(test_return.dims[2].second, a));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {k, j, i}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i, j, k}));
}

TEST(TensorCopyNodeTest, permutation_3d_equal_dims) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(2);
    auto c = symbolic::integer(4);
    math::tensor::TensorLayout layout_x({a, b, c});
    math::tensor::TensorLayout layout_y({a, c, b});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 3);
    auto i = test_return.dims[0].first;
    auto j = test_return.dims[1].first;
    auto k = test_return.dims[2].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(symbolic::eq(test_return.dims[1].second, c));
    ASSERT_TRUE(symbolic::eq(test_return.dims[2].second, b));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {i, k, j}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i, j, k}));
}

TEST(TensorCopyNodeTest, unsqueeze) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(3);
    auto one = symbolic::one();
    auto zero = symbolic::zero();
    math::tensor::TensorLayout layout_x({a, b});
    math::tensor::TensorLayout layout_y({a, one, b});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 2);
    auto i = test_return.dims[0].first;
    auto j = test_return.dims[1].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(symbolic::eq(test_return.dims[1].second, b));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {i, j}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i, zero, j}));
}

TEST(TensorCopyNodeTest, unsqueeze_multiple) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(3);
    auto one = symbolic::one();
    auto zero = symbolic::zero();
    math::tensor::TensorLayout layout_x({a, b});
    math::tensor::TensorLayout layout_y({one, a, one, b, one});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 2);
    auto i = test_return.dims[0].first;
    auto j = test_return.dims[1].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(symbolic::eq(test_return.dims[1].second, b));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {i, j}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {zero, i, zero, j, zero}));
}

TEST(TensorCopyNodeTest, squeeze) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(3);
    auto one = symbolic::one();
    auto zero = symbolic::zero();
    math::tensor::TensorLayout layout_x({a, one, b});
    math::tensor::TensorLayout layout_y({a, b});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 2);
    auto i = test_return.dims[0].first;
    auto j = test_return.dims[1].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(symbolic::eq(test_return.dims[1].second, b));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {i, zero, j}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i, j}));
}

TEST(TensorCopyNodeTest, squeeze_multiple) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(3);
    auto one = symbolic::one();
    auto zero = symbolic::zero();
    math::tensor::TensorLayout layout_x({one, a, one, b, one});
    math::tensor::TensorLayout layout_y({a, b});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 2);
    auto i = test_return.dims[0].first;
    auto j = test_return.dims[1].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(symbolic::eq(test_return.dims[1].second, b));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {zero, i, zero, j, zero}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i, j}));
}

TEST(TensorCopyNodeTest, reshape_to_one) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(3);
    auto c = symbolic::integer(6);
    math::tensor::TensorLayout layout_x({a, b});
    math::tensor::TensorLayout layout_y({c});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 1);
    auto i = test_return.dims[0].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, c));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {symbolic::div(i, b), symbolic::mod(i, b)}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i}));
}

TEST(TensorCopyNodeTest, reshape_to_one_front) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(3);
    auto c = symbolic::integer(6);
    auto one = symbolic::one();
    math::tensor::TensorLayout layout_x({a, b});
    math::tensor::TensorLayout layout_y({c, one});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 1);
    auto i = test_return.dims[0].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, c));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {symbolic::div(i, b), symbolic::mod(i, b)}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i, symbolic::zero()}));
}

TEST(TensorCopyNodeTest, reshape_to_one_back) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(3);
    auto c = symbolic::integer(6);
    auto one = symbolic::one();
    math::tensor::TensorLayout layout_x({a, b});
    math::tensor::TensorLayout layout_y({one, c});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 1);
    auto i = test_return.dims[0].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, c));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {symbolic::div(i, b), symbolic::mod(i, b)}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {symbolic::div(i, c), symbolic::mod(i, c)}));
}

TEST(TensorCopyNodeTest, reshape_from_one) {
    auto a = symbolic::integer(6);
    auto b = symbolic::integer(2);
    auto c = symbolic::integer(3);
    math::tensor::TensorLayout layout_x({a});
    math::tensor::TensorLayout layout_y({b, c});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 1);
    auto i = test_return.dims[0].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {i}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {symbolic::div(i, c), symbolic::mod(i, c)}));
}

TEST(TensorCopyNodeTest, reshape_from_one_front) {
    auto a = symbolic::integer(6);
    auto b = symbolic::integer(2);
    auto c = symbolic::integer(3);
    auto one = symbolic::one();
    math::tensor::TensorLayout layout_x({a, one});
    math::tensor::TensorLayout layout_y({b, c});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 1);
    auto i = test_return.dims[0].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {i, symbolic::zero()}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {symbolic::div(i, c), symbolic::mod(i, c)}));
}

TEST(TensorCopyNodeTest, reshape_from_one_back) {
    auto a = symbolic::integer(6);
    auto b = symbolic::integer(2);
    auto c = symbolic::integer(3);
    auto one = symbolic::one();
    math::tensor::TensorLayout layout_x({one, a});
    math::tensor::TensorLayout layout_y({b, c});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 1);
    auto i = test_return.dims[0].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {symbolic::div(i, a), symbolic::mod(i, a)}));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {symbolic::div(i, c), symbolic::mod(i, c)}));
}

TEST(TensorCopyNodeTest, reshape_to_one_big) {
    auto a = symbolic::integer(2);
    auto b = symbolic::integer(4);
    auto c = symbolic::integer(3);
    auto d = symbolic::integer(2);
    auto e = symbolic::integer(48);
    math::tensor::TensorLayout layout_x({a, b, c, d});
    math::tensor::TensorLayout layout_y({e});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 1);
    auto i = test_return.dims[0].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, e));
    ASSERT_TRUE(subsets_eq(
        test_return.subset_x,
        {symbolic::div(i, symbolic::integer(24)),
         symbolic::mod(symbolic::div(i, symbolic::integer(6)), b),
         symbolic::mod(symbolic::div(i, d), c),
         symbolic::mod(i, d)}
    ));
    ASSERT_TRUE(subsets_eq(test_return.subset_y, {i}));
}

TEST(TensorCopyNodeTest, reshape_from_one_big) {
    auto a = symbolic::integer(48);
    auto b = symbolic::integer(2);
    auto c = symbolic::integer(4);
    auto d = symbolic::integer(3);
    auto e = symbolic::integer(2);
    math::tensor::TensorLayout layout_x({a});
    math::tensor::TensorLayout layout_y({b, c, d, e});
    TensorCopyNodeTestReturn test_return;
    create_and_test(layout_x, layout_y, test_return);
    ASSERT_EQ(test_return.dims.size(), 1);
    auto i = test_return.dims[0].first;
    ASSERT_TRUE(symbolic::eq(test_return.dims[0].second, a));
    ASSERT_TRUE(subsets_eq(test_return.subset_x, {i}));
    ASSERT_TRUE(subsets_eq(
        test_return.subset_y,
        {symbolic::div(i, symbolic::integer(24)),
         symbolic::mod(symbolic::div(i, symbolic::integer(6)), c),
         symbolic::mod(symbolic::div(i, e), d),
         symbolic::mod(i, e)}
    ));
}
