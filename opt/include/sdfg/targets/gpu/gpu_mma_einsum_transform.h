#pragma once

#include "sdfg/einsum/einsum_node.h"
#include "sdfg/einsum/replacers/einsum2matmul.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg::gpu {

class GpuMmaEinsumReplacer : public einsum::Einsum2MatMul {
    const GpuArch* arch_;
    int wave_tile_m_;
    int wave_tile_n_;

public:
    /// @p wave_tile_m x @p wave_tile_n MMA blocks per wave (register blocking).
    GpuMmaEinsumReplacer(const GpuArch* arch, int wave_tile_m = 1, int wave_tile_n = 1);

    struct EinsumMmaAnalysis : public MatMulAnalysis {};

    bool analyze(const einsum::EinsumCluster& cluster, EinsumMmaAnalysis& result) const;

    einsum::ReplaceOutcome
    replace(einsum::EinsumReplacementContext& context, const einsum::EinsumCluster& cluster) const override;

    bool can_be_applied(const einsum::EinsumCluster& cluster) const override;

protected:
    bool matches_possible_mma_pattern(const MatMulAnalysis& analysis) const;
    bool wave_tile_divides(const MatMulAnalysis& analysis) const;
};

class GpuMmaEinsumTransform : public transformations::Transformation {
    bool matched_ = false;
    StructuredLoop& outermost_mma_loop_;
    const gpu::GpuArch* arch_;
    int wave_tile_m_;
    int wave_tile_n_;

protected:
    /// the einsum parts modify the SDFG in place, so WILL ALWAYS CHANGE IT. We need to revert the changes if we did not
    /// get sth. we liked
    bool run_internal(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool verify_only
    );

public:
    GpuMmaEinsumTransform(
        StructuredLoop& outermoost_mma_loop, const gpu::GpuArch* arch = nullptr, int wave_tile_m = 1, int wave_tile_n = 1
    );

    std::string name() const override {
        return "GpuMmaEinsumTransform";
    }

    bool try_apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    bool matched() const {
        return matched_;
    }
};

} // namespace sdfg::gpu
