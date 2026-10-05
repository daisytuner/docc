#pragma once

#include <string>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/rocm/rocm_arch.h"
#include "sdfg/types/type.h"

namespace sdfg::rocm {

class ROCMLanguageExtension : public sdfg::codegen::LanguageExtension {
    const gpu::rocm::RocmArch* arch_;

public:
    ROCMLanguageExtension(
        sdfg::Function& function, const gpu::rocm::RocmArch* arch, const std::string& external_prefix = ""
    )
        : LanguageExtension(function, external_prefix), arch_(arch) {
    }

    ROCMLanguageExtension(sdfg::Function& function, const std::string& external_prefix = "")
        : ROCMLanguageExtension(function, nullptr, external_prefix) {
    }

    const std::string language() const override {
        return "ROCM";
    }

    const gpu::rocm::RocmArch* gpu_arch() const {
        return arch_;
    }

    std::string primitive_type(const types::PrimitiveType prim_type) override;

    std::string declaration(
        const std::string& name, const types::IType& type, bool use_initializer = false, bool use_alignment = false
    ) override;

    std::string type_cast(const std::string& name, const types::IType& type) override;

    std::string subset(const types::IType& type, const data_flow::Subset& subset) override;

    std::string expression(const symbolic::Expression expr) override;

    std::string access_node(const data_flow::AccessNode& node) override;

    std::string tasklet(const data_flow::Tasklet& tasklet) override;

    std::string zero(const types::PrimitiveType prim_type) override;
};

} // namespace sdfg::rocm
