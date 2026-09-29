#include <string>

#include <sdfg/data_flow/library_nodes/math/tensor/matmul_node.h>
#include <sdfg/passes/expansion/library_node_expansion_pass.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/targets/rocm/rocm_arch.h>
#include <sdfg/transformations/transformation.h>
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/control_flow_node.h"


namespace sdfg::gpu::rocm {

// Transformation wrapper around the ROCm MMA (rocwmma) library-node expander.
// Given a MatMul library node and a target arch, it expands the node into the
// arch-specific MMA implementation, mirroring the C++ ``expand_single_node``.
class RocmMmaTransform : public transformations::Transformation {
    sdfg::math::tensor::MatMulNode& node_;
    const sdfg::gpu::rocm::RocmArch* arch_;
    bool expanded_ = false;

public:
    RocmMmaTransform(sdfg::math::tensor::MatMulNode& node, const sdfg::gpu::rocm::RocmArch* arch);

    std::string name() const override;

    bool can_be_applied(sdfg::builder::StructuredSDFGBuilder&, sdfg::analysis::AnalysisManager&) override;

    void apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager&) override;

    bool expanded() const;

    void to_json(nlohmann::json& j) const override;

    static RocmMmaTransform from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc);
};

} // namespace sdfg::gpu::rocm
