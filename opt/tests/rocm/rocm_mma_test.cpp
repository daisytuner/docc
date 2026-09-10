#include "sdfg/targets/rocm/rocm_mma.h"

#include <gtest/gtest.h>
#include <sdfg/serializer/json_serializer.h>
#include <strstream>

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_types.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_arch.h"
#include "sdfg/types/tensor.h"
#include "sdfg_debug_dump.h"

namespace test::utils {

struct TestCodeSnippet {
    std::string name;
    bool as_file;
    std::string extension;
    std::string content;
};

struct TestCodegenOut {
    std::string main_function;
    std::unordered_map<std::string, TestCodeSnippet> snippets;

    TestCodegenOut(
        const std::string& main_function, const std::unordered_map<std::string, sdfg::codegen::CodeSnippet>& snippets
    )
        : main_function(main_function) {
        for (const auto& [name, snippet] : snippets) {
            this->snippets[name] = TestCodeSnippet{
                .name = name,
                .as_file = snippet.is_as_file(),
                .extension = snippet.extension(),
                .content = snippet.stream().str()
            };
        }
    }
};

TestCodegenOut test_codegen(sdfg::StructuredSDFG& sdfg, const std::string& group = "code", bool dump = true) {
    sdfg::analysis::AnalysisManager ana(sdfg);
    auto instr_plan = sdfg::codegen::InstrumentationPlan::none(sdfg);
    auto cap_plan = sdfg::codegen::ArgCapturePlan::none(sdfg);
    auto snippetFactory = std::make_shared<sdfg::codegen::CodeSnippetFactory>();
    snippetFactory->add_available_dependency(sdfg::gpu::rocm::RocmWmmaLibDependency::instance());
    sdfg::codegen::CPPCodeGenerator codegen(sdfg, ana, *instr_plan, *cap_plan, snippetFactory);
    codegen.generate();
    std::stringstream ss;
    codegen.append_function_source(ss);

    if (dump) {
        if (auto dir = get_test_output_dir()) {
            auto out_dir = dir.value() / group;
            std::filesystem::create_directories(out_dir);
            auto main_file = out_dir / "main.cpp";
            std::ofstream ofs(main_file.c_str(), std::ofstream::out);
            ofs << ss.str();
            ofs.close();
            for (auto& [name, snippet] : codegen.library_snippets()) {
                auto snippet_file = out_dir / (name + "." + snippet.extension());
                std::ofstream ofs_snippet(snippet_file.c_str(), std::ofstream::out);
                ofs_snippet << snippet.stream().str();
                ofs_snippet.close();
            }
        }
    }

    return TestCodegenOut(ss.str(), codegen.library_snippets());
}

} // namespace test::utils

namespace sdfg::rocm {

TEST(ROCMMMATest, ScopedExpansion) {
    // C[512, 512] = A[512, 1024] @ B[1024, 512]
    constexpr int M = 1024; // rows of A / C
    constexpr int N = 1024; // cols of B / C
    constexpr int K = 1024; // contraction dimension
    constexpr int TILE = 16; // MMA-friendly tile width for the outer maps

    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_desc(types::PrimitiveType::Half);
    types::Pointer dev_pointer_type(base_desc);
    dev_pointer_type.storage_type().value("AMD_Generic");
    types::Scalar index_desc(types::PrimitiveType::Int32);
    index_desc.storage_type().value("AMD_Generic");

    // Matrix pointers (device buffers) and the map induction variables.
    builder.add_container("A", dev_pointer_type, true);
    builder.add_container("B", dev_pointer_type, true);
    builder.add_container("C", dev_pointer_type, true);
    builder.add_container("row", index_desc);
    builder.add_container("col", index_desc);

    auto row = symbolic::symbol("row");
    auto col = symbolic::symbol("col");

    // Outer map over the rows of C: i = 0, TILE, 2*TILE, ...
    auto& map_row = builder.add_map(
        root,
        row,
        symbolic::Lt(row, symbolic::integer(M)),
        symbolic::integer(0),
        symbolic::add(row, symbolic::integer(TILE)),
        gpu::ScheduleType_GPU_Offload::create<ScheduleType_ROCM_Offload>(gpu::TargetLevel::Y_GRID, symbolic::integer(M))
    );

    // Inner map over the columns of C: j = 0, TILE, 2*TILE, ...
    auto& map_col = builder.add_map(
        map_row.root(),
        col,
        symbolic::Lt(col, symbolic::integer(N)),
        symbolic::integer(0),
        symbolic::add(col, symbolic::integer(TILE)),
        gpu::ScheduleType_GPU_Offload::create<ScheduleType_ROCM_Offload>(gpu::TargetLevel::X_GRID, symbolic::integer(N))
    );

    auto& block = builder.add_block(map_col.root());

    auto& a_node = builder.add_access(block, "A");
    auto& b_node = builder.add_access(block, "B");
    auto& c_node = builder.add_access(block, "C");

    // Slice layouts for this (i, j) tile. Strides follow the row-major layout of the
    // full matrices, while the offset selects the tile inside the base pointer.
    //   A tile: [TILE, K]  starting at row i           -> offset i * K
    //   B tile: [K, TILE]  starting at column j        -> offset j
    //   C tile: [TILE, TILE] starting at (i, j)        -> offset i * N + j
    math::tensor::TensorLayout a_layout(
        {symbolic::integer(TILE), symbolic::integer(K)},
        {symbolic::integer(K), symbolic::integer(1)},
        symbolic::mul(row, symbolic::integer(K))
    );
    math::tensor::TensorLayout
        b_layout({symbolic::integer(K), symbolic::integer(TILE)}, {symbolic::integer(N), symbolic::integer(1)}, col);
    math::tensor::TensorLayout c_layout(
        {symbolic::integer(TILE), symbolic::integer(TILE)},
        {symbolic::integer(N), symbolic::integer(1)},
        symbolic::add(symbolic::mul(row, symbolic::integer(N)), col)
    );

    types::Tensor a_tensor(base_desc, a_layout);
    types::Tensor b_tensor(base_desc, b_layout);
    types::Tensor c_tensor(base_desc, c_layout);

    auto& matmul_node = static_cast<math::tensor::MatMulNode&>(builder.add_library_node<math::tensor::MatMulNode>(
        block, DebugInfo(), a_layout, b_layout, types::PrimitiveType::Half, &c_layout
    ));

    builder.add_computational_memlet(block, a_node, matmul_node, "A", {}, a_tensor, block.debug_info());
    builder.add_computational_memlet(block, b_node, matmul_node, "B", {}, b_tensor, block.debug_info());
    builder.add_computational_memlet(block, c_node, matmul_node, "Y", {}, c_tensor, block.debug_info());

    dump_sdfg(builder.subject(), "0.init");

    EXPECT_NO_THROW(sdfg.validate());

    passes::expansion::
        expand_single_node(builder, block, matmul_node, gpu::rocm::RocmMmaExpander(gpu::rocm::ROCM_ARCH_GFX1201));

    dump_sdfg(builder.subject(), "1.expanded");

    EXPECT_NO_THROW(sdfg.validate());

    test::utils::test_codegen(sdfg, "result", true);
}


} // namespace sdfg::rocm
