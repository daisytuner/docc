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
    int k_block_multiple_ = 1;

public:
    GpuMmaEinsumReplacer(
        const GpuArch* arch,
        const MmaBlockSize* block_size_hint = nullptr,
        const BlockCounts* wave_tile_hint = nullptr,
        int k_block_multiple = 1
    );

    struct EinsumMmaAnalysis : public MatMulAnalysis {
        types::PrimitiveType acc_type;
        /// Some extent is not a constant multiple of the MMA block: the tiling covers the padded extents and the
        /// fragments crossing the exact extents are clipped.
        bool ragged = false;
    };

    bool analyze(const einsum::EinsumCluster& cluster, EinsumMmaAnalysis& result) const;

    einsum::ReplaceOutcome
    replace(einsum::EinsumReplacementContext& context, const einsum::EinsumCluster& cluster) const override;

    bool can_be_applied(const einsum::EinsumCluster& cluster) const override;

protected:
    bool matches_possible_mma_pattern(EinsumMmaAnalysis& analysis) const;

    /// The MMA tiling for @p analysis: exact if the extents divide, else (if the target supports it) ragged.
    std::optional<GpuMmaTiling> select_tiling(EinsumMmaAnalysis& analysis) const;
};

class GpuMmaEinsumTransform : public transformations::Transformation {
    bool matched_ = false;
    StructuredLoop& outermost_mma_loop_;
    const gpu::GpuArch* arch_;

    std::optional<MmaBlockSize> block_size_hint_;
    std::optional<BlockCounts> wave_tiling_hint_;
    /// Pads a ragged K-block loop to a multiple of this (e.g. the K-panel size in blocks), so a later panel
    /// tiling divides it evenly. The padded blocks are clipped, i.e. read as zeros.
    int k_block_multiple_ = 1;

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
        const BlockCounts* wave_tiling_hint = nullptr,
        int k_block_multiple = 1
    );

    /// Owning overload: takes the hints by value and stores them internally, so
    /// the caller does not have to keep separate hint objects alive. Intended
    /// for the backwards-compatible Python constructor, which synthesises the
    /// hints from scalar arguments. An empty optional means "no hint".
    GpuMmaEinsumTransform(
        StructuredLoop& outermoost_mma_loop,
        const gpu::GpuArch* arch,
        std::optional<MmaBlockSize> block_size_hint,
        std::optional<BlockCounts> wave_tile_hint,
        int k_block_multiple = 1
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
