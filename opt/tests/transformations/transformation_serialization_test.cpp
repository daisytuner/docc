#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/symbolic/symbolic.h>

#include <sdfg/transformations/loop_distribute.h>
#include <sdfg/transformations/loop_interchange.h>
#include <sdfg/transformations/loop_skewing.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/transformations/out_local_storage.h>

#include <sdfg/transformations/offloading/cuda_parallelize_nested_map.h>
#include <sdfg/transformations/offloading/cuda_transform.h>
#include <sdfg/transformations/offloading/gpu_condition_propagation.h>
#include <sdfg/transformations/offloading/gpu_loop_reordering.h>
#include <sdfg/transformations/offloading/gpu_tiling.h>
#include <sdfg/transformations/offloading/kernel_local_storage.h>
#include <sdfg/transformations/omp_transform.h>

#ifdef DOCC_HAS_TARGET_TENSTORRENT
#include <docc/target/tenstorrent/tenstorrent_transform.h>
#endif

using namespace sdfg;

namespace {

void ValidateSerialization(const nlohmann::json& j, std::size_t expected_subgraph_size) {
    ASSERT_TRUE(j.contains("transformation_type"));
    ASSERT_TRUE(j["transformation_type"].is_string());

    ASSERT_TRUE(j.contains("subgraph"));
    ASSERT_TRUE(j["subgraph"].is_object());
    ASSERT_EQ(j["subgraph"].size(), expected_subgraph_size);

    for (std::size_t i = 0; i < expected_subgraph_size; ++i) {
        auto key = std::to_string(i);
        ASSERT_TRUE(j["subgraph"].contains(key));
        const auto& node = j["subgraph"][key];
        ASSERT_TRUE(node.contains("element_id"));
        ASSERT_TRUE(node["element_id"].is_number_unsigned());
        ASSERT_TRUE(node.contains("type"));
        ASSERT_TRUE(node["type"].is_string());
    }
}

// Build a minimal SDFG with one map nest i->j suitable for most loop-based transforms.
struct LoopFixture {
    builder::StructuredSDFGBuilder builder;
    structured_control_flow::Map* outer_map;
    structured_control_flow::Map* inner_map;
    data_flow::AccessNode* access_A;

    LoopFixture()
        : builder("serialization_test", FunctionType_CPU), outer_map(nullptr), inner_map(nullptr), access_A(nullptr) {
        auto& root = builder.subject().root();

        auto bound = symbolic::integer(16);
        auto indvar_i = symbolic::symbol("i");
        outer_map = &builder.add_map(
            root,
            indvar_i,
            symbolic::Lt(indvar_i, bound),
            symbolic::integer(0),
            symbolic::add(indvar_i, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create()
        );

        auto& body = outer_map->root();
        auto indvar_j = symbolic::symbol("j");
        inner_map = &builder.add_map(
            body,
            indvar_j,
            symbolic::Lt(indvar_j, bound),
            symbolic::integer(0),
            symbolic::add(indvar_j, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create()
        );

        // One container for OutLocalStorage / KernelLocalStorage
        types::Scalar base_desc(types::PrimitiveType::Float);
        types::Array arr_desc(base_desc, symbolic::integer(16));
        types::Pointer ptr_desc(arr_desc);
        builder.add_container("A", ptr_desc, true);

        // Add a block with an access node for "A" inside the inner map
        auto& inner_body = inner_map->root();
        auto& block = builder.add_block(inner_body);
        access_A = &builder.add_access(block, "A");
    }
};

TEST(TransformationSerializationTest, CoreLoopTransformationsShape) {
    LoopFixture f;

    // LoopTiling
    transformations::LoopTiling tiling(*f.outer_map, 4);
    nlohmann::json j;
    tiling.to_json(j);
    ValidateSerialization(j, 1);

    auto tiling2 = transformations::LoopTiling::from_json(f.builder, j);
    ASSERT_EQ(tiling2.name(), tiling.name());

    // LoopDistribute
    transformations::LoopDistribute distribute(*f.outer_map);
    nlohmann::json jd;
    distribute.to_json(jd);
    ValidateSerialization(jd, 1);
    auto distribute2 = transformations::LoopDistribute::from_json(f.builder, jd);
    ASSERT_EQ(distribute2.name(), distribute.name());

    // LoopInterchange
    transformations::LoopInterchange interchange(*f.outer_map, *f.inner_map);
    nlohmann::json ji;
    interchange.to_json(ji);
    ValidateSerialization(ji, 2);
    auto interchange2 = transformations::LoopInterchange::from_json(f.builder, ji);
    ASSERT_EQ(interchange2.name(), interchange.name());

    // OutLocalStorage
    transformations::OutLocalStorage ols(*f.outer_map, *f.access_A);
    nlohmann::json jo;
    ols.to_json(jo);
    ValidateSerialization(jo, 2);
    auto ols2 = transformations::OutLocalStorage::from_json(f.builder, jo);
    ASSERT_EQ(ols2.name(), ols.name());

    // LoopSkewing
    transformations::LoopSkewing skew(*f.outer_map, *f.inner_map, 1);
    nlohmann::json js;
    skew.to_json(js);
    ValidateSerialization(js, 2);
    auto skew2 = transformations::LoopSkewing::from_json(f.builder, js);
    ASSERT_EQ(skew2.name(), skew.name());
}

TEST(TransformationSerializationTest, OffloadingAndGPUTransformationsShape) {
    LoopFixture f;

    // CUDATransform
    cuda::CUDATransform cuda_t(*f.outer_map, 32);
    nlohmann::json jc;
    cuda_t.to_json(jc);
    ValidateSerialization(jc, 1);
    auto cuda_t2 = cuda::CUDATransform::from_json(f.builder, jc);
    ASSERT_EQ(cuda_t2.name(), cuda_t.name());

    // CUDAParallelizeNestedMap
    transformations::CUDAParallelizeNestedMap nested(*f.inner_map, 32);
    nlohmann::json jn;
    nested.to_json(jn);
    ValidateSerialization(jn, 1);
    auto nested2 = transformations::CUDAParallelizeNestedMap::from_json(f.builder, jn);
    ASSERT_EQ(nested2.name(), nested.name());

    // GPUTiling
    transformations::GPUTiling gpu_tiling(*static_cast<structured_control_flow::StructuredLoop*>(f.outer_map), 4);
    nlohmann::json jgt;
    gpu_tiling.to_json(jgt);
    ValidateSerialization(jgt, 1);
    auto gpu_tiling2 = transformations::GPUTiling::from_json(f.builder, jgt);
    ASSERT_EQ(gpu_tiling2.name(), gpu_tiling.name());

    // KernelLocalStorage
    symbolic::Expression offset = symbolic::integer(0);
    transformations::KernelLocalStorage
        kls(*static_cast<structured_control_flow::StructuredLoop*>(f.outer_map), offset, *f.access_A);
    nlohmann::json jkls;
    kls.to_json(jkls);
    ValidateSerialization(jkls, 2);
    auto kls2 = transformations::KernelLocalStorage::from_json(f.builder, jkls);
    ASSERT_EQ(kls2.name(), kls.name());

    // GPULoopReordering
    transformations::GPULoopReordering reordering(*f.outer_map);
    nlohmann::json jr;
    reordering.to_json(jr);
    ValidateSerialization(jr, 1);
    auto reordering2 = transformations::GPULoopReordering::from_json(f.builder, jr);
    ASSERT_EQ(reordering2.name(), reordering.name());

    // GPUConditionPropagation
    transformations::GPUConditionPropagation cond_prop(*f.outer_map);
    nlohmann::json jcp;
    cond_prop.to_json(jcp);
    ValidateSerialization(jcp, 1);
    auto cond_prop2 = transformations::GPUConditionPropagation::from_json(f.builder, jcp);
    ASSERT_EQ(cond_prop2.name(), cond_prop.name());
}

TEST(TransformationSerializationTest, OtherScheduleTransformationsShape) {
    LoopFixture f;

    // OMPTransform
    transformations::OMPTransform omp_t(*f.outer_map);
    nlohmann::json jo;
    omp_t.to_json(jo);
    ValidateSerialization(jo, 1);
    auto omp_t2 = transformations::OMPTransform::from_json(f.builder, jo);
    ASSERT_EQ(omp_t2.name(), omp_t.name());

#ifdef DOCC_HAS_TARGET_TENSTORRENT
    // TenstorrentTransform
    sdfg::analysis::AnalysisManager analysis_manager(f.builder.subject());
    tenstorrent::TenstorrentTransform tt_t(f.builder, analysis_manager, *f.outer_map);
    nlohmann::json jtt;
    tt_t.to_json(jtt);
    ValidateSerialization(jtt, 1);
    auto tt_t2 = tenstorrent::TenstorrentTransform::from_json(f.builder, analysis_manager, jtt);
    ASSERT_EQ(tt_t2.name(), tt_t.name());
#endif
}

} // namespace
