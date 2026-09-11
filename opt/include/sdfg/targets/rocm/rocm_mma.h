#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/targets/gpu/gpu_mma.h"
#include "sdfg/targets/rocm/rocm_arch.h"

namespace sdfg::gpu::rocm {

inline data_flow::ImplementationType ImplementationType_ROCM_MMA("ROCM_MMA");

class RocmMmaExpander : public GpuMmaExpander {
    const RocmArch& arch_;

protected:
    GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const override;
    bool matches_possible_mma_pattern(const math::tensor::MatMulNode& node) const override;
    ScheduleType get_schedule_type(gpu::TargetLevel dim, const symbolic::Integer& size) const override;
    void set_implementation_type_mma(math::tensor::MatMulNode& node, const GpuMmaTiling& mma_tiling) const override;

public:
    RocmMmaExpander(const RocmArch& arch) : GpuMmaExpander(), arch_(arch) {}
};

class RocmMmaMatmulDispatcher : public GpuMmaMatmulDispatcher {
protected:
    const GpuMmaSupport* get_mma_arch() const override;

public:
    RocmMmaMatmulDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::tensor::MatMulNode& node
    )
        : GpuMmaMatmulDispatcher(language_extension, function, data_flow_graph, node) {}

    static GpuMmaTiling get_mma_tiling(const GpuMmaSupport* arch, const symbolic::MultiExpression& res_shape);

protected:
    void emit_block_frag_declaration(
        codegen::CodegenOutput& out,
        const std::string& name,
        FragmentType type,
        std::array<int, 3> dims,
        math::tensor::TensorLayout::TensorLayoutType layout,
        types::PrimitiveType scalar_type,
        std::optional<std::pair<int, int>> coop_dims = std::nullopt
    ) const override;

    GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const override;

    void emit_load_macro(
        codegen::CodegenOutput& out,
        const std::string& name,
        const std::string& base_addr,
        const symbolic::Expression& offset,
        const symbolic::Expression& line_size,
        math::tensor::TensorLayout::TensorLayoutType layout
    ) const override;
    void emit_store_macro(
        const codegen::CodegenOutput& out,
        const std::string& frag_name,
        const std::string& base_addr,
        const symbolic::Expression& offset,
        const symbolic::Expression& line_size,
        math::tensor::TensorLayout::TensorLayoutType layout
    ) const override;

    void emit_frag_zero_init(codegen::CodegenOutput& out, const std::string& frag_name, types::PrimitiveType scalar_type)
        const override;
    void emit_mma_compute(
        codegen::CodegenOutput& out,
        const std::string& frag_d_out,
        const std::string& frag_a,
        const std::string& frag_b,
        const std::string& frag_c_in
    ) const override;
};

class RocmWmmaLibDependency : public codegen::LibDependency {
public:
    static const RocmWmmaLibDependency* instance() {
        static RocmWmmaLibDependency inst;
        return &inst;
    }

    std::string_view name() const override { return "rocwmma"; }
    void enumerate_includes(std::vector<std::string>& out_list) const override {
        out_list.push_back("rocwmma/rocwmma.hpp");
    }
    std::vector<std::string_view>& globally_unique_ids() const override {
        static std::vector<std::string_view> ids{"rocwmma"};
        return ids;
    }
};

} // namespace sdfg::gpu::rocm
