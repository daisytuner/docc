#pragma once

#include "sdfg/einsum/einsum_node.h"
#include "sdfg/einsum/einsum_state.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg::gpu {

class GpuMmaEinsumTransform : public transformations::Transformation {
    bool matched_ = false;
    StructuredLoop& outermost_mma_loop_;
    const gpu::GpuArch* arch_;

protected:
    void try_map_to_mma_einsum(
        const builder::StructuredSDFGBuilder& builder,
        const analysis::AnalysisManager& analysis_manager,
        einsum::EinsumNode* einsum_node
    );

    /// the einsum parts modify the SDFG in place, so WILL ALWAYS CHANGE IT. We need to revert the changes if we did not
    /// get sth. we liked
    bool run_internal(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool revert_changes
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
