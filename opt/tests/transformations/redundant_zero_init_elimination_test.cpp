#include "sdfg/transformations/redundant_zero_init_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fill_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_load_node.h"
#include "sdfg/targets/gpu/gpu_mma_fragment_store_node.h"
#include "sdfg/targets/gpu/gpu_mma_matmul_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/transformations/transformation_registry.h"
#include "sdfg/visitor/for_each.h"

using namespace sdfg;

namespace {

constexpr int N = 16;
const gpu::MmaBlockSize kBlock{16, 16, 16};

struct Options {
    bool annotated = true;
    std::string init_value = "0.0";
    bool fused_init_body = false; ///< an additional read in the init block
    bool block_between = false; ///< an unrelated block between the init and the MMA nest
    std::vector<std::string> transfers_between; ///< containers transferred by offloading blocks in between
    bool assignment_between = false; ///< a symbol assignment block in between
    bool with_mma = true;
    bool store_to_other = false; ///< the MMA stores to a different container than the init writes
    bool extra_read_of_y = false; ///< the MMA nest reads Y other than through a C-fragment load
    bool c_load_in_for = false; ///< the C-fragment load sits in a sequential loop
};

struct Built {
    structured_control_flow::Map* init = nullptr;
    structured_control_flow::StructuredLoop* mma_nest = nullptr;
};

void add_frag_load(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const std::string& frag,
    const std::string& memory,
    gpu::MmaFragmentType frag_type
) {
    types::Pointer half_ptr{types::Scalar(types::PrimitiveType::Half)};
    auto& load = builder.add_library_node<gpu::GpuMmaFragmentLoadNode>(
        block,
        DebugInfo(),
        kBlock,
        frag_type,
        gpu::GpuMmaFromMemoryLayout{
            .offset = symbolic::zero(), .ldstride = symbolic::integer(N), .layout = gpu::MMA_LAYOUT_ROW_MAJOR
        },
        types::PrimitiveType::Half,
        data_flow::ImplementationType_NONE
    );
    builder.add_computational_memlet(
        block,
        builder.add_access(block, frag),
        load,
        load.input(gpu::GpuMmaFragmentLoadNode::FRAG_INPUT_IDX),
        {},
        half_ptr
    );
    builder.add_computational_memlet(
        block,
        builder.add_access(block, memory),
        load,
        load.input(gpu::GpuMmaFragmentLoadNode::PTR_INPUT_IDX),
        {},
        half_ptr
    );
}

/// Root: [init map i, j: Y[i*N + j] = 0]  [MMA map w: fill(acc); for k: load a, b; mma; load c <- Y; store acc -> Y]
Built build(builder::StructuredSDFGBuilder& builder, const Options& opts = {}) {
    auto& root = builder.subject().root();
    types::Scalar half(types::PrimitiveType::Half);
    types::Pointer half_ptr(half);
    types::Scalar idx(types::PrimitiveType::Int32);
    for (auto* name : {"A", "B", "Y", "Z"}) {
        builder.add_container(name, half_ptr, true);
        builder.add_container(std::string(name) + "_host", half_ptr, true);
    }
    for (auto* name : {"acc", "frag_a", "frag_b", "frag_c"}) {
        builder.add_container(name, half_ptr);
    }
    for (auto* name : {"i", "j", "w", "k", "s"}) {
        builder.add_container(name, idx);
    }
    auto sym = [](const char* name) {
        return symbolic::symbol(name);
    };
    auto bounded = [&](const char* name, int bound) {
        return symbolic::Lt(sym(name), symbolic::integer(bound));
    };
    auto inc = [&](const char* name) {
        return symbolic::add(sym(name), symbolic::one());
    };

    Built built;

    // ---- init nest ----
    auto schedule = structured_control_flow::ScheduleType_Sequential::create();
    if (opts.annotated) {
        structured_control_flow::ScheduleType_ZeroInit::set(schedule);
    }
    built.init = &builder.add_map(root, sym("i"), bounded("i", N), symbolic::zero(), inc("i"), schedule);
    auto& j_map = builder.add_map(
        built.init->root(),
        sym("j"),
        bounded("j", N),
        symbolic::zero(),
        inc("j"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    {
        auto& block = builder.add_block(j_map.root());
        auto& value = builder.add_constant(block, opts.init_value, half);
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(block, value, tasklet, "_in", {}, half);
        auto y_idx = symbolic::add(symbolic::mul(sym("i"), symbolic::integer(N)), sym("j"));
        builder.add_computational_memlet(block, tasklet, "_out", builder.add_access(block, "Y"), {y_idx}, half_ptr);
        if (opts.fused_init_body) {
            auto& t2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
            builder.add_computational_memlet(block, builder.add_access(block, "A"), t2, "_in", {y_idx}, half_ptr);
            builder.add_computational_memlet(block, t2, "_out", builder.add_access(block, "B"), {y_idx}, half_ptr);
        }
    }

    if (opts.block_between) {
        builder.add_block(root);
    }
    for (auto& container : opts.transfers_between) {
        offloading::add_offloading_block<rocm::ROCMDataOffloadingNode>(
            builder,
            root,
            container + "_host",
            container,
            offloading::DataTransferDirection::NONE,
            offloading::BufferLifecycle::ALLOC,
            half_ptr,
            DebugInfo(),
            symbolic::integer(N * N * 2),
            symbolic::integer(0)
        );
    }
    if (opts.assignment_between) {
        builder.add_assignments(root, {{symbolic::symbol("s"), symbolic::zero()}});
    }

    // ---- MMA nest ----
    auto& w_map = builder.add_map(
        root,
        sym("w"),
        bounded("w", 1),
        symbolic::zero(),
        inc("w"),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    built.mma_nest = &w_map;
    {
        auto& block = builder.add_block(w_map.root());
        auto& fill = builder.add_library_node<gpu::GpuMmaFillNode>(
            block, DebugInfo(), kBlock, types::PrimitiveType::Half, data_flow::ImplementationType_NONE
        );
        builder.add_computational_memlet(block, builder.add_access(block, "acc"), fill, fill.input(0), {}, half_ptr);
    }
    auto& k_loop = builder.add_for(w_map.root(), sym("k"), bounded("k", 1), symbolic::zero(), inc("k"));
    {
        auto& block = builder.add_block(k_loop.root());
        add_frag_load(builder, block, "frag_a", "A", gpu::MmaFragmentType::A);
        add_frag_load(builder, block, "frag_b", "B", gpu::MmaFragmentType::B);
    }
    if (opts.with_mma) {
        auto& block = builder.add_block(k_loop.root());
        auto& mma = builder.add_library_node<gpu::GpuMmaMatmulNode>(
            block,
            DebugInfo(),
            kBlock,
            types::PrimitiveType::Half,
            types::PrimitiveType::Half,
            data_flow::ImplementationType_NONE
        );
        builder.add_computational_memlet(
            block, builder.add_access(block, "acc"), mma, mma.input(gpu::GpuMmaMatmulNode::Y_INPUT_IDX), {}, half_ptr
        );
        builder.add_computational_memlet(
            block, builder.add_access(block, "frag_a"), mma, mma.input(gpu::GpuMmaMatmulNode::A_INPUT_IDX), {}, half_ptr
        );
        builder.add_computational_memlet(
            block, builder.add_access(block, "frag_b"), mma, mma.input(gpu::GpuMmaMatmulNode::B_INPUT_IDX), {}, half_ptr
        );
    }
    {
        structured_control_flow::Sequence* c_scope = &w_map.root();
        if (opts.c_load_in_for) {
            c_scope = &builder.add_for(*c_scope, sym("s"), bounded("s", 2), symbolic::zero(), inc("s")).root();
        }
        auto& block = builder.add_block(*c_scope);
        add_frag_load(builder, block, "frag_c", "Y", gpu::MmaFragmentType::C);
    }
    if (opts.extra_read_of_y) {
        auto& block = builder.add_block(w_map.root());
        auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder
            .add_computational_memlet(block, builder.add_access(block, "Y"), tasklet, "_in", {symbolic::zero()}, half_ptr);
        builder
            .add_computational_memlet(block, tasklet, "_out", builder.add_access(block, "Z"), {symbolic::zero()}, half_ptr);
    }
    {
        auto& block = builder.add_block(w_map.root());
        auto& store = builder.add_library_node<gpu::GpuMmaFragmentStoreNode>(
            block,
            DebugInfo(),
            kBlock,
            gpu::MmaFragmentType::C,
            gpu::GpuMmaFromMemoryLayout{
                .offset = symbolic::zero(), .ldstride = symbolic::integer(N), .layout = gpu::MMA_LAYOUT_ROW_MAJOR
            },
            types::PrimitiveType::Half,
            data_flow::ImplementationType_NONE
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, "acc"),
            store,
            store.input(gpu::GpuMmaFragmentStoreNode::FRAG_INPUT_IDX),
            {},
            half_ptr
        );
        builder.add_computational_memlet(
            block,
            builder.add_access(block, opts.store_to_other ? "Z" : "Y"),
            store,
            store.input(gpu::GpuMmaFragmentStoreNode::PTR_INPUT_IDX),
            {},
            half_ptr
        );
    }
    return built;
}

struct Counts {
    size_t fills = 0;
    size_t c_loads = 0;
    size_t stores = 0;
};

Counts count(structured_control_flow::Sequence& root) {
    Counts counts;
    visitor::for_each_block(root, [&](structured_control_flow::Block& block) {
        for (auto* lib : block.dataflow().library_nodes()) {
            if (dynamic_cast<gpu::GpuMmaFillNode*>(lib)) {
                ++counts.fills;
            } else if (auto* load = dynamic_cast<gpu::GpuMmaFragmentLoadNode*>(lib)) {
                counts.c_loads += load->fragment_type() == gpu::MmaFragmentType::C;
            } else if (dynamic_cast<gpu::GpuMmaFragmentStoreNode*>(lib)) {
                ++counts.stores;
            }
        }
    });
    return counts;
}

bool applicable(const Options& opts) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto built = build(builder, opts);
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::RedundantZeroInitElimination transformation(*built.init);
    return transformation.can_be_applied(builder, analysis_manager);
}

} // namespace

TEST(RedundantZeroInitEliminationTest, RemovesInitAndReplacesCLoadWithFill) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto built = build(builder);
    auto& root = builder.subject().root();
    EXPECT_NO_THROW(builder.subject().validate());
    ASSERT_EQ(root.size(), 2);
    auto before = count(root);
    EXPECT_EQ(before.fills, 1);
    EXPECT_EQ(before.c_loads, 1);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::RedundantZeroInitElimination transformation(*built.init);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(builder.subject().validate());

    ASSERT_EQ(root.size(), 1);
    EXPECT_EQ(&root.at(0), built.mma_nest);
    auto after = count(root);
    EXPECT_EQ(after.fills, 2);
    EXPECT_EQ(after.c_loads, 0);
    EXPECT_EQ(after.stores, 1);

    // The C fragment is now zero-filled and Y is only stored to.
    bool c_filled = false;
    size_t y_accesses = 0;
    visitor::for_each_block(root, [&](structured_control_flow::Block& block) {
        for (auto* lib : block.dataflow().library_nodes()) {
            if (auto* fill = dynamic_cast<gpu::GpuMmaFillNode*>(lib)) {
                auto& edge = *block.dataflow().in_edges(*fill).begin();
                if (static_cast<data_flow::AccessNode&>(edge.src()).data() == "frag_c") {
                    c_filled = true;
                    EXPECT_EQ(fill->fill_type(), types::PrimitiveType::Half);
                    EXPECT_EQ(fill->mma_block_size().m, kBlock.m);
                }
            }
        }
        for (auto* access : block.dataflow().data_nodes()) {
            y_accesses += access->data() == "Y";
        }
    });
    EXPECT_TRUE(c_filled);
    EXPECT_EQ(y_accesses, 1);
}

TEST(RedundantZeroInitEliminationTest, Positive) {
    EXPECT_TRUE(applicable({}));
}

TEST(RedundantZeroInitEliminationTest, NotAnnotated) {
    EXPECT_FALSE(applicable({.annotated = false}));
}

TEST(RedundantZeroInitEliminationTest, IntegerAndFloatZeroLiterals) {
    EXPECT_TRUE(applicable({.init_value = "0"}));
    EXPECT_TRUE(applicable({.init_value = "0.0f"}));
}

TEST(RedundantZeroInitEliminationTest, NonZeroInit) {
    EXPECT_FALSE(applicable({.init_value = "1.0"}));
}

TEST(RedundantZeroInitEliminationTest, FusedInitBody) {
    EXPECT_FALSE(applicable({.fused_init_body = true}));
}

TEST(RedundantZeroInitEliminationTest, NotAdjacent) {
    EXPECT_FALSE(applicable({.block_between = true}));
}

TEST(RedundantZeroInitEliminationTest, SkipsUnrelatedTransferBlocks) {
    EXPECT_TRUE(applicable({.transfers_between = {"Z"}}));
    EXPECT_TRUE(applicable({.transfers_between = {"A", "B", "Z"}}));
}

TEST(RedundantZeroInitEliminationTest, TransferOfOutputInBetween) {
    // A transfer of Y between the nests could change what the MMA reads or what the host observes.
    EXPECT_FALSE(applicable({.transfers_between = {"Y"}}));
    EXPECT_FALSE(applicable({.transfers_between = {"A", "Y"}}));
}

TEST(RedundantZeroInitEliminationTest, AssignmentInBetween) {
    EXPECT_FALSE(applicable({.assignment_between = true}));
}

TEST(RedundantZeroInitEliminationTest, RemovesInitAndKeepsTransferBlocks) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto built = build(builder, {.transfers_between = {"A", "B"}});
    auto& root = builder.subject().root();
    ASSERT_EQ(root.size(), 4);
    auto* transfer_a = &root.at(1);
    auto* transfer_b = &root.at(2);

    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::RedundantZeroInitElimination transformation(*built.init);
    ASSERT_TRUE(transformation.can_be_applied(builder, analysis_manager));
    transformation.apply(builder, analysis_manager);
    EXPECT_NO_THROW(builder.subject().validate());

    ASSERT_EQ(root.size(), 3);
    EXPECT_EQ(&root.at(0), transfer_a);
    EXPECT_EQ(&root.at(1), transfer_b);
    EXPECT_EQ(&root.at(2), built.mma_nest);
    auto after = count(root);
    EXPECT_EQ(after.c_loads, 0);
    EXPECT_EQ(after.fills, 2);
}

TEST(RedundantZeroInitEliminationTest, NextNestIsNotMma) {
    EXPECT_FALSE(applicable({.with_mma = false}));
}

TEST(RedundantZeroInitEliminationTest, MmaStoresOtherContainer) {
    EXPECT_FALSE(applicable({.store_to_other = true}));
}

TEST(RedundantZeroInitEliminationTest, OtherReadOfOutput) {
    EXPECT_FALSE(applicable({.extra_read_of_y = true}));
}

TEST(RedundantZeroInitEliminationTest, CLoadInSequentialLoop) {
    EXPECT_FALSE(applicable({.c_load_in_for = true}));
}

TEST(RedundantZeroInitEliminationTest, InitIsLastChild) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto built = build(builder);
    builder.remove_child(builder.subject().root(), 1);
    analysis::AnalysisManager analysis_manager(builder.subject());
    transformations::RedundantZeroInitElimination transformation(*built.init);
    EXPECT_FALSE(transformation.can_be_applied(builder, analysis_manager));
}

TEST(RedundantZeroInitEliminationTest, Serialization) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);
    auto built = build(builder);

    transformations::RedundantZeroInitElimination transformation(*built.init);
    nlohmann::json j;
    transformation.to_json(j);
    EXPECT_EQ(j["transformation_type"], "RedundantZeroInitElimination");
    EXPECT_EQ(j["subgraph"]["0"]["element_id"], built.init->element_id());

    std::string error;
    // Dispatches through the transformation registry.
    EXPECT_TRUE(transformations::verify_transformation_round_trip(builder, j, error)) << error;

    auto restored = transformations::RedundantZeroInitElimination::from_json(builder, j);
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(restored.can_be_applied(builder, analysis_manager));
}
