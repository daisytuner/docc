#pragma once

#include <string>
#include <vector>
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/scheduler/loop_scheduler.h"
#include "sdfg/passes/scheduler/sdfg_scheduler.h"

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

/// How a frontend runs a target's artifact.
enum class ArtifactKind { SharedLibrary, PythonModule };

/// The parameters of a compile request that a target's compile hook receives.
struct CodegenRequest {
    std::string output_folder;
    std::string instrumentation_mode;
    bool capture_args = false;
    bool debug_build = false;
    int threads = 0;
    bool reuse_sources = false;
    const sdfg::Options* options = &sdfg::Options::empty();
};

/**
 * What a target generates in place of docc's C++ shared library, if it has its own code generator.
 *
 * Frontends written in Python may run a Python build step after compiling and load the artifact with a Python loader,
 * both named as "package.module:function".
 */
struct TargetArtifact {
    /// File the compile hook writes, "{name}" standing for the SDFG's name, e.g. "{name}.axs.py".
    std::string file;
    ArtifactKind kind = ArtifactKind::SharedLibrary;
    /// Python callable (artifact_path, sdfg_name) -> callable running the artifact; empty loads it by its kind.
    std::string loader;
    /// Python callable (artifact_path, sdfg_name) run after compiling, e.g. an external toolchain; empty for none.
    std::string build;
    /// Writes the artifact into the request's output folder and returns its path.
    std::string (*compile)(sdfg::StructuredSDFG& sdfg, const CodegenRequest& request) = nullptr;
};

/// The precision a target computes in: of untyped literals, of floating-point contractions (see fp_math), and which
/// float types it stores.
struct TargetPrecision {
    /// Types of untyped numeric literals, as the Python frontends type them; not of explicit dtypes or arguments.
    std::string literal_float = "float64";
    std::string literal_int = "int64";
    /// How Float and Double matmuls compute when the user selects nothing, as TargetOptions::contraction_precision
    /// spells it, e.g. an Ozaki emulation for a target that multiplies only integers.
    std::string default_contraction = "native";
    /// Whether quantized and Ozaki contractions multiply by their scales and compute in the contraction's own type, as
    /// a target without Double, floor or exact powers of two needs (fp_math::Rescaling::Direct); Ozaki then slices.
    bool direct_rescaling = false;
    /// The significand bits, guard bits included, in which the target computes the results of emulated contractions,
    /// e.g. 13 for a unit computing in a float of 11 bits; Ozaki slices no operand finer. 0 for the result type's own.
    int contraction_bits = 0;
    /// Float types the target does not store, such as bfloat16 on a unit without it, besides FP8, which no code
    /// generator stores: after the contraction precision, which slices by the operands' own types,
    /// passes::FloatWidening turns them into Float, and the runtime converts the arguments of these types at the
    /// boundary.
    std::vector<sdfg::types::PrimitiveType> widened_float_types = {};
};

/// How a target schedules an SDFG, in pipeline order; run by sdfg::scheduling::schedule. Every target sets both.
struct TargetScheduling {
    using SDFGSchedulerFactory = std::unique_ptr<sdfg::passes::scheduler::SDFGScheduler> (*)(const TargetOptions&);
    using LoopSchedulersFactory =
        std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> (*)(const TargetOptions&);

    /// Brings the whole SDFG into the target's normal form, before remote tuning.
    SDFGSchedulerFactory create_sdfg_scheduler = nullptr;
    /// The loop schedulers run after remote tuning, possibly none.
    LoopSchedulersFactory create_loop_schedulers = nullptr;
    /// Whether remote tuning asks the server to normalize, for targets whose SDFG scheduler does not.
    bool remote_normalization = false;
};

struct DoccTarget {
    static constexpr int NEWEST_API_VER = 6;

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

    // Since API version 5, the target's properties are grouped by the stage reading them, in pipeline order. Plugins
    // of earlier versions get the defaults.

    /// Whether docc lifts contractions from loops into einsums before expansion, when the user leaves einsum unset.
    bool lift_einsums = false;

    [[nodiscard]] bool safe_get_lift_einsums() const {
        return api_ver >= 5 && lift_einsums;
    }

    /// How contractions compute and which float types the target stores.
    TargetPrecision precision;

    [[nodiscard]] const TargetPrecision& safe_get_precision() const {
        static const TargetPrecision defaults;
        return api_ver >= 5 ? precision : defaults;
    }

    /// How the target schedules an SDFG. Since API version 6, required.
    TargetScheduling scheduling;

    /// The target's own artifact; without a compile hook, docc generates and builds C++.
    TargetArtifact artifact;

    [[nodiscard]] const TargetArtifact* safe_get_artifact() const {
        return api_ver >= 5 && artifact.compile ? &artifact : nullptr;
    }
};

} // namespace docc::target
