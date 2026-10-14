#pragma once

#include "sdfg/einsum/einsum_node.h"
#include "sdfg/einsum/replacers/einsum2matmul.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg::gpu {

class GpuMmaEinsumReplacer : public einsum::Einsum2MatMul {
    const GpuArch* arch_;
    const BlockCounts* wave_tile_hint_ = nullptr;
    const MmaBlockSize* block_size_hint_ = nullptr;

public:
    GpuMmaEinsumReplacer(
        const GpuArch* arch, const MmaBlockSize* block_size_hint = nullptr, const BlockCounts* wave_tile_hint = nullptr
    );

    struct EinsumMmaAnalysis : public MatMulAnalysis {
        types::PrimitiveType acc_type;
    };

    bool analyze(const einsum::EinsumCluster& cluster, EinsumMmaAnalysis& result) const;

    einsum::ReplaceOutcome
    replace(einsum::EinsumReplacementContext& context, const einsum::EinsumCluster& cluster) const override;

    bool can_be_applied(const einsum::EinsumCluster& cluster) const override;

protected:
    bool matches_possible_mma_pattern(EinsumMmaAnalysis& analysis) const;
};

class GpuMmaEinsumTransform : public transformations::Transformation {
    bool matched_ = false;
    StructuredLoop& outermost_mma_loop_;
    const gpu::GpuArch* arch_;

    std::optional<MmaBlockSize> block_size_hint_;
    std::optional<BlockCounts> wave_tiling_hint_;

protected:
    /// the einsum parts modify the SDFG in place, so WILL ALWAYS CHANGE IT. We need to revert the changes if we did not
    /// get sth. we liked
    bool run_internal(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool verify_only
    );

public:
    GpuMmaEinsumTransform(
        StructuredLoop& outermoost_mma_loop,
        const gpu::GpuArch* arch = nullptr,
        const MmaBlockSize* block_size_hint = nullptr,
        const BlockCounts* wave_tiling_hint = nullptr
    );

    /// Owning overload: takes the hints by value and stores them internally, so
    /// the caller does not have to keep separate hint objects alive. Intended
    /// for the backwards-compatible Python constructor, which synthesises the
    /// hints from scalar arguments. An empty optional means "no hint".
    GpuMmaEinsumTransform(
        StructuredLoop& outermoost_mma_loop,
        const gpu::GpuArch* arch,
        std::optional<MmaBlockSize> block_size_hint,
        std::optional<BlockCounts> wave_tile_hint
    );

    std::string name() const override {
        return "GpuMmaEinsumTransform";
    }

    bool try_apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static GpuMmaEinsumTransform from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    bool matched() const {
        return matched_;
    }
};

} // namespace sdfg::gpu
