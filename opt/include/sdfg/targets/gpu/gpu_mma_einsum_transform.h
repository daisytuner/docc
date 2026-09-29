#pragma once

#include "sdfg/einsum/einsum_node.h"
#include "sdfg/einsum/einsum_state.h"
#include "sdfg/einsum/replacers/einsum2matmul.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg::gpu {

class GpuMmaEinsumReplacer : public einsum::Einsum2MatMul {
    const GpuArch* arch_;

public:
    GpuMmaEinsumReplacer(const GpuArch* arch);

    struct EinsumMmaAnalysis : public MatMulAnalysis {};

    bool analyze(const einsum::EinsumCluster& cluster, EinsumMmaAnalysis& result) const;

    einsum::ReplaceOutcome replace(einsum::EinsumReplacementContext& context, const einsum::EinsumCluster& cluster)
        const override;

    bool can_be_applied(const einsum::EinsumCluster& cluster) const override;

protected:
    bool matches_possible_mma_pattern(const MatMulAnalysis& analysis) const;
};

class GpuMmaEinsumTransform : public transformations::Transformation {
    bool matched_ = false;
    StructuredLoop& outermost_mma_loop_;
    const gpu::GpuArch* arch_;

protected:
    /// the einsum parts modify the SDFG in place, so WILL ALWAYS CHANGE IT. We need to revert the changes if we did not
    /// get sth. we liked
    bool run_internal(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool verify_only
    );

public:
    GpuMmaEinsumTransform(StructuredLoop& outermoost_mma_loop, const gpu::GpuArch* arch = nullptr);

    std::string name() const override { return "GpuMmaEinsumTransform"; }

    bool try_apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    bool matched() const { return matched_; }
};

} // namespace sdfg::gpu
