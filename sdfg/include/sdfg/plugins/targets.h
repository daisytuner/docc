#pragma once

#include <string>
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/scheduler/loop_scheduler.h"

namespace docc::compile {
class SrcFileCompilerBuilder;
}

namespace docc::target {

struct TargetOptions {
    std::string target;
    std::string category;
    bool remote_tuning;
    bool already_normalized = false;
    bool enable_fusion_in_normalize = true;
    bool use_new_fusion_in_simplify = true;
    bool fp_contract = true;
    /// How Float and Double matmuls compute: native, int2 ... int8 (quantized), ozaki, ozaki1[:s] or ozaki2[:L]
    /// (emulated on Int8 matmuls); see fp_math::ContractionPrecision.
    std::string contraction_precision = "native";
    // Non-owning reference to the caller's option bag (typically owned by the
    // front-end / PyStructuredSDFG). Stored as a pointer so it can be reseated
    // after default construction (e.g. filled field-by-field from Python).
    const sdfg::Options* options = &sdfg::Options::empty();
};

struct DoccTarget {
    static constexpr int NEWEST_API_VER = 3;

    int api_ver = NEWEST_API_VER;
    std::string short_name;

    bool (*apply_additional_compile_options)(docc::compile::SrcFileCompilerBuilder& src_compile_builder);
    /**
     * Consider these hooks experimental. They allow completely custom modification of the SDFG.
     * In the future we may want to move more towards pass-management
     */

    bool (*apply_expand_time_mapping)(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        const TargetOptions& options
    );
    bool (*apply_sched_time_mapping)(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        const TargetOptions& options
    );
    std::function<bool(docc::compile::SrcFileCompilerBuilder& src_compile_builder)>
    safe_apply_additional_compile_options_fn_get() {
        if (api_ver >= 1) {
            return apply_additional_compile_options;
        } else {
            return nullptr;
        }
    }
    std::function<bool(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        const TargetOptions& options
    )>
    safe_apply_expand_time_mapping_fn_get() {
        if (api_ver >= 1) {
            return apply_expand_time_mapping;
        } else {
            return nullptr;
        }
    }
    std::function<bool(
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::analysis::AnalysisManager& analysis_manager,
        const TargetOptions& options
    )>
    safe_apply_sched_time_mapping_fn_get() {
        if (api_ver >= 1) {
            return apply_sched_time_mapping;
        } else {
            return nullptr;
        }
    }

    std::vector<std::shared_ptr<
        sdfg::passes::scheduler::LoopScheduler>> (*get_target_loop_schedulers)(const TargetOptions& options);
    [[nodiscard]] std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>>
    safe_get_target_loop_schedulers(const TargetOptions& options) const {
        if (api_ver >= 2 && get_target_loop_schedulers != nullptr) {
            return get_target_loop_schedulers(options);
        } else {
            return {};
        }
    }

    /// How Float and Double matmuls compute when the user selects nothing, as TargetOptions::contraction_precision
    /// spells it, e.g. an Ozaki emulation for a target that multiplies only integers.
    std::string default_contraction_precision = "native";

    [[nodiscard]] std::string safe_get_default_contraction_precision() const {
        return api_ver >= 3 ? default_contraction_precision : "native";
    }

    /// Whether quantized and Ozaki contractions multiply by their scales and compute in the contraction's own type, as
    /// a target without Double, floor or exact powers of two needs (fp_math::Rescaling::Direct); Ozaki then slices.
    bool direct_contraction_rescaling = false;

    [[nodiscard]] bool safe_get_direct_contraction_rescaling() const {
        return api_ver >= 3 && direct_contraction_rescaling;
    }
};

} // namespace docc::target
