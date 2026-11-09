#pragma once

#include <string>

#include "sdfg/codegen/language_extensions/c_style_language_extension.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/rocm/rocm_arch.h"
#include "sdfg/types/type.h"

namespace sdfg::rocm {

class ROCMLanguageExtension : public sdfg::codegen::CStyleLanguageExtension {
    const gpu::rocm::RocmArch* arch_;

public:
    ROCMLanguageExtension(
        sdfg::Function& function, const gpu::rocm::RocmArch* arch, const std::string& external_prefix = ""
    )
        : CStyleLanguageExtension(function, external_prefix), arch_(arch) {
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

    /// hip_bfloat16 has no native arithmetic and rounds after every operator; compute in float.
    types::PrimitiveType compute_type(const types::PrimitiveType prim_type) override {
        return prim_type == types::PrimitiveType::BFloat ? types::PrimitiveType::Float : prim_type;
    }

    /// A nontemporal load. The builtin needs a native vector type; HIP's int2/int4 are structs.
    std::string streaming_copy(const std::string& dst, const std::string& src, size_t bytes) override {
        if (bytes != 4 && bytes != 8 && bytes != 16) {
            return "";
        }
        const std::string type = "__docc_nt" + std::to_string(bytes);
        const std::string vector = bytes == 4 ? ""
                                              : " __attribute__((ext_vector_type(" + std::to_string(bytes / 4) + ")))";
        return "{ typedef int " + type + vector + "; *reinterpret_cast<" + type + "*>(" + dst +
               ") = __builtin_nontemporal_load(reinterpret_cast<const " + type + "*>(" + src + ")); }";
    }
};

} // namespace sdfg::rocm
