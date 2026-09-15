#include "sdfg/targets/rocm/rocm_mma_transform.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_mma_expander.h"

namespace sdfg::gpu::rocm {

RocmMmaTransform::RocmMmaTransform(sdfg::math::tensor::MatMulNode& node, const sdfg::gpu::rocm::RocmArch* arch)
    : node_(node), arch_(arch) {}

std::string RocmMmaTransform::name() const { return "RocmMmaTransform"; }

bool RocmMmaTransform::can_be_applied(sdfg::builder::StructuredSDFGBuilder&, sdfg::analysis::AnalysisManager&) {
    // Search the parents up until we find a map with the ROCm offloaded schedule type.
    auto& dataflow = node_.get_parent();
    ControlFlowNode* scope = static_cast<structured_control_flow::Block*>(dataflow.get_parent());
    bool in_rocm_map = false;
    while (scope != nullptr) {
        if (auto* map = dyn_cast<structured_control_flow::Map*>(scope)) {
            if (map->schedule_type().value() == ::sdfg::rocm::ScheduleType_ROCM_Offload::value()) {
                in_rocm_map = true;
                break;
            }
        }
        scope = scope->get_parent();
    }
    if (!in_rocm_map) {
        return false;
    }
    RocmMmaExpander expander(*arch_);
    return expander.for_lib_node(node_) != nullptr;
}

void RocmMmaTransform::apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager&) {
    sdfg::gpu::rocm::RocmMmaExpander expander(*arch_);
    auto& dataflow = node_.get_parent();
    auto& block = static_cast<Block&>(*dataflow.get_parent());
    auto outcome = sdfg::passes::expansion::expand_single_node(builder, block, node_, expander);
    expanded_ = outcome.expanded;
}

bool RocmMmaTransform::expanded() const { return expanded_; }

void RocmMmaTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = name();
    j["arch"] = arch_->name();
    j["matmul_node"] = node_.element_id();
}

RocmMmaTransform RocmMmaTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto arch_name = desc.at("arch").get<std::string>();
    const sdfg::gpu::rocm::RocmArch* arch = sdfg::gpu::rocm::rocm_arch_parse(arch_name);
    if (!arch) {
        throw transformations::InvalidTransformationDescriptionException("Unsupported ROCm architecture: " + arch_name);
    }
    auto node_id = desc.at("matmul_node").get<size_t>();
    auto* elem = builder.find_element_by_id(node_id);
    if (elem == nullptr) {
        throw transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(node_id) + " not found.");
    }
    auto* node = dyn_cast<math::tensor::MatMulNode*>(elem);
    if (node == nullptr) {
        throw transformations::InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(node_id) + " is not a MatMulNode."
        );
    }
    return RocmMmaTransform(*node, arch);
}

} // namespace sdfg::gpu::rocm
