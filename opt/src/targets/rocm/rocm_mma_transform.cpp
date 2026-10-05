#include "sdfg/targets/rocm/rocm_mma_transform.h"
#include "sdfg/targets/gpu/gpu_mma_expander.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg::gpu::rocm {

GpuMmaTransform::GpuMmaTransform(sdfg::math::tensor::MatMulNode& node, const GpuArch* arch) : node_(node), arch_(arch) {
}

std::string GpuMmaTransform::name() const {
    return "GpuMmaTransform";
}

bool GpuMmaTransform::can_be_applied(sdfg::builder::StructuredSDFGBuilder&, sdfg::analysis::AnalysisManager&) {
    // Search the parents up until we find a map with the ROCm offloaded schedule type.
    auto& dataflow = node_.get_parent();
    ControlFlowNode* scope = static_cast<structured_control_flow::Block*>(dataflow.get_parent());
    bool in_gpu_arch_map = false;
    while (scope != nullptr) {
        if (auto* map = dyn_cast<structured_control_flow::Map*>(scope)) {
            if (auto* arch = GpuArch::get_from_schedule_type(map->schedule_type())) {
                in_gpu_arch_map = true;
                if (!this->arch_) {
                    this->arch_ = arch;
                }
                break;
            }
        }
        scope = scope->get_parent();
    }
    if (!in_gpu_arch_map) {
        return false;
    }
    GpuMmaExpander expander(arch_);
    return expander.for_lib_node(node_) != nullptr;
}

void GpuMmaTransform::apply(sdfg::builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager&) {
    GpuMmaExpander expander(arch_);
    auto& dataflow = node_.get_parent();
    auto& block = static_cast<Block&>(*dataflow.get_parent());
    auto outcome = sdfg::passes::expansion::expand_single_node(builder, block, node_, expander);
    expanded_ = outcome.expanded;
}

bool GpuMmaTransform::expanded() const {
    return expanded_;
}

void GpuMmaTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = name();
    j["arch"] = arch_->name();
    j["matmul_node"] = node_.element_id();
}

GpuMmaTransform GpuMmaTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    const GpuArch* arch;
    auto it = desc.find("arch");
    if (it != desc.end()) {
        std::string arch_name = it->get<std::string>();
        arch = GpuArch::parse_from_name(arch_name);
        if (!arch) {
            throw transformations::InvalidTransformationDescriptionException("Unsupported GPU architecture: " + arch_name);
        }
    } else {
        arch = nullptr;
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
    return GpuMmaTransform(*node, arch);
}

} // namespace sdfg::gpu::rocm
