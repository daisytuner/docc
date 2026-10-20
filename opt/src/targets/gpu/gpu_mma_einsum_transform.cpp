#include <algorithm>
#include <ranges>

#include "sdfg/targets/gpu/gpu_mma_einsum_transform.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/einsum/einsum_detection.h"
#include "sdfg/einsum/einsum_replacer.h"
#include "sdfg/einsum/passes/einsum_passes.h"
#include "sdfg/einsum/replacers/einsum2matmul.h"
#include "sdfg/einsum/transformations/einsum_lift.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/targets/gpu/gpu_mma_expander.h"
#include "sdfg/visitor/structured_sdfg_walker.h"

namespace sdfg::gpu {

GpuMmaEinsumReplacer::GpuMmaEinsumReplacer(
    const GpuArch* arch, const MmaBlockSize* block_size_hint, const BlockCounts* wave_tile_hint, int k_block_multiple
)
    : arch_(arch), block_size_hint_(block_size_hint), wave_tile_hint_(wave_tile_hint),
      k_block_multiple_(std::max(1, k_block_multiple)) {
}

std::optional<GpuMmaTiling> GpuMmaEinsumReplacer::select_tiling(EinsumMmaAnalysis& analysis) const {
    auto* mma_arch = arch_->mma_support();
    analysis.ragged = false;
    if (auto exact = mma_arch->get_mma_tiling(
            {analysis.m, analysis.n, analysis.k},
            analysis.input_type,
            analysis.acc_type,
            block_size_hint_,
            wave_tile_hint_
        )) {
        return exact;
    }
    // Ragged extents (e.g. the `min(i0 + T, N) - i0` of a non-dividing tile, or K not a block multiple): tile the
    // constant over-approximation and clip the fragments crossing the exact extent.
    auto pad = [](const symbolic::Expression& e) {
        return symbolic::simplify(symbolic::expand(symbolic::overapproximate(e)));
    };
    auto ragged = mma_arch->get_ragged_mma_tiling(
        {pad(analysis.m), pad(analysis.n), pad(analysis.k)},
        analysis.input_type,
        analysis.output_type,
        analysis.acc_type,
        block_size_hint_,
        wave_tile_hint_
    );
    analysis.ragged = ragged.has_value();
    return ragged;
}

bool GpuMmaEinsumReplacer::matches_possible_mma_pattern(EinsumMmaAnalysis& analysis) const {
    if (!arch_) {
        return false;
    }

    auto* mma_arch = arch_->mma_support();
    if (!mma_arch) {
        return false;
    }

    // basic sanity checks
    auto& dims_a = analysis.layout_a.value();
    auto& dims_b = analysis.layout_b.value();
    if (dims_a.dims() != 2 || dims_b.dims() != 2) {
        return false;
    }

    if (!symbolic::eq(dims_a.get_dim(1), dims_b.get_dim(0))) {
        // K dimension must match
        return false;
    }

    auto layout_a = dims_a.is_2d_col_or_row_major();
    auto layout_b = dims_b.is_2d_col_or_row_major();
    if (layout_a == math::tensor::TensorLayout::LAYOUT_OTHER || layout_b == math::tensor::TensorLayout::LAYOUT_OTHER ||
        analysis.layout_y.value().is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_OTHER) {
        // only support row-major or col-major layout
        return false;
    }

    auto input_type = analysis.input_type;
    auto output_type = analysis.output_type;

    if (!mma_arch->supported_types(input_type, output_type)) {
        return false;
    }

    auto acc_type = mma_arch->get_accumulator_type(input_type, output_type, types::PrimitiveType::Void);
    analysis.acc_type = acc_type;

    return select_tiling(analysis).has_value();
}

bool GpuMmaEinsumReplacer::analyze(const einsum::EinsumCluster& cluster, EinsumMmaAnalysis& result) const {
    if (Einsum2MatMul::analyze(cluster, result)) {
        return matches_possible_mma_pattern(result);
    } else {
        return false;
    }
}

einsum::ReplaceOutcome GpuMmaEinsumReplacer::
    replace(einsum::EinsumReplacementContext& context, const einsum::EinsumCluster& cluster) const {
    using Dir = passes::LibNodeExpander::InputUse;

    // --- Applicability (mirrors Einsum2Gemm, restricted to the canonical MatMul form) ---

    EinsumMmaAnalysis analysis;
    if (!this->analyze(cluster, analysis)) {
        return context.unapplicable();
    }

    auto mma_tiling = select_tiling(analysis);

    if (!mma_tiling) {
        return context.unable();
    }

    // --- Replacement ---

    std::vector<Dir> access_dirs{Dir::Scalar, Dir::Scalar, Dir::Scalar}; // last one is the output ptr
    auto standalone = context.replacement_requires_access_nodes(access_dirs, true);
    if (!standalone) {
        return context.unable();
    }

    return GpuMmaExpander::expand_mma_standalone(
        *standalone,
        *arch_,
        mma_tiling.value(),
        analysis.layout_a.value(),
        analysis.layout_b.value(),
        analysis.layout_y.value(),
        analysis.input_type,
        analysis.output_type,
        true,
        cluster.consumed_loops.front()->debug_info(),
        // {y, a, b}: the cluster lists the output after its inputs, and A/B in no particular order.
        {static_cast<int>(cluster.inputs.size()), analysis.a_idx, analysis.b_idx},
        analysis.ragged,
        k_block_multiple_
    );
}

bool GpuMmaEinsumReplacer::can_be_applied(const einsum::EinsumCluster& cluster) const {
    EinsumMmaAnalysis analysis;
    return this->analyze(cluster, analysis);
}

bool GpuMmaEinsumTransform::run_internal(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool verify_only
) {
    auto& sdfg = builder.subject();

    if (!arch_) {
        auto loop_stack = ControlFlowNode::parent_chain(outermost_mma_loop_);
        for (auto entry : std::views::reverse(loop_stack)) {
            if (auto* map = dyn_cast<StructuredLoop*>(entry)) {
                if (map->schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader) {
                    arch_ = GpuArch::get_from_schedule_type(map->schedule_type());
                    break;
                }
            }
        }
    }

    einsum::EinsumDetection detector;
    detector.run(outermost_mma_loop_);

    std::vector<einsum::EinsumNode*> mapped_einsums;
    for (auto& einsum_node : detector.einsums()) {
        GpuMmaEinsumReplacer repl(
            arch_,
            block_size_hint_ ? &*block_size_hint_ : nullptr,
            wave_tiling_hint_ ? &*wave_tiling_hint_ : nullptr,
            k_block_multiple_
        );
        // einsum::Einsum2MatMul repl;
        if (verify_only) {
            matched_ |= repl.can_be_applied(*einsum_node);
        } else {
            auto outcome = einsum::replace_einsum_cluster(builder, *einsum_node, repl);
            if (outcome.applied) {
                matched_ = true;
            }
        }
    }

    return matched_;
}

GpuMmaEinsumTransform::GpuMmaEinsumTransform(
    StructuredLoop& outermoost_mma_loop,
    const gpu::GpuArch* arch,
    const MmaBlockSize* block_size_hint,
    const BlockCounts* wave_tiling_hint,
    int k_block_multiple
)
    : outermost_mma_loop_(outermoost_mma_loop), arch_(arch),
      block_size_hint_(block_size_hint ? std::optional<MmaBlockSize>(*block_size_hint) : std::nullopt),
      wave_tiling_hint_(wave_tiling_hint ? std::optional<BlockCounts>(*wave_tiling_hint) : std::nullopt),
      k_block_multiple_(std::max(1, k_block_multiple)) {
    // Copy any provided hints into our own storage so we never retain pointers
    // into caller-owned memory (which would dangle once the caller returns).
}

GpuMmaEinsumTransform::GpuMmaEinsumTransform(
    StructuredLoop& outermoost_mma_loop,
    const gpu::GpuArch* arch,
    std::optional<MmaBlockSize> block_size_hint,
    std::optional<BlockCounts> wave_tile_hint,
    int k_block_multiple
)
    : outermost_mma_loop_(outermoost_mma_loop), arch_(arch), block_size_hint_(block_size_hint),
      wave_tiling_hint_(wave_tile_hint), k_block_multiple_(std::max(1, k_block_multiple)) {
    // Hints are stored by value, so copying/moving this transform stays safe.
}

bool GpuMmaEinsumTransform::try_apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return run_internal(builder, analysis_manager, false);
}

void GpuMmaEinsumTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto result = run_internal(builder, analysis_manager, false);
    if (!result) {
        throw std::runtime_error("GpuMmaEinsumTransform failed to apply");
    }
}

bool GpuMmaEinsumTransform::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return run_internal(builder, analysis_manager, true);
}

void GpuMmaEinsumTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    if (arch_) {
        j["parameters"]["arch"] = arch_->unique_id();
    }
    if (block_size_hint_) {
        j["parameters"]["block_size"]["m"] = block_size_hint_->m;
        j["parameters"]["block_size"]["n"] = block_size_hint_->n;
        j["parameters"]["block_size"]["k"] = block_size_hint_->k;
    }
    if (wave_tiling_hint_) {
        j["parameters"]["wave_tile"]["blocks_m"] = wave_tiling_hint_->blocks_m;
        j["parameters"]["wave_tile"]["blocks_n"] = wave_tiling_hint_->blocks_n;
    }
    if (k_block_multiple_ > 1) {
        j["parameters"]["k_block_multiple"] = k_block_multiple_;
    }

    serializer::JSONSerializer serializer(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["0"], outermost_mma_loop_);
}

GpuMmaEinsumTransform GpuMmaEinsumTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto loop_id = j.at("subgraph").at("0").at("element_id").get<size_t>();
    auto* element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto* loop = dyn_cast<StructuredLoop*>(element);
    if (!loop) {
        throw transformations::InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop."
        );
    }

    const auto& params = j.value("parameters", nlohmann::json::object());

    const GpuArch* arch = nullptr;
    if (params.contains("arch")) {
        std::string arch_name = params.at("arch").get<std::string>();
        arch = GpuArch::parse_from_name(arch_name);
        if (!arch) {
            throw transformations::InvalidTransformationDescriptionException("Unsupported GPU architecture: " + arch_name);
        }
    }

    std::optional<MmaBlockSize> block_size;
    if (params.contains("block_size")) {
        const auto& bs = params.at("block_size");
        block_size = MmaBlockSize{bs.at("m").get<int>(), bs.at("n").get<int>(), bs.at("k").get<int>()};
    }

    std::optional<BlockCounts> wave_tile;
    if (params.contains("wave_tile")) {
        const auto& wt = params.at("wave_tile");
        wave_tile = BlockCounts{wt.at("blocks_m").get<int>(), wt.at("blocks_n").get<int>()};
    }

    return GpuMmaEinsumTransform(*loop, arch, block_size, wave_tile, params.value("k_block_multiple", 1));
}

} // namespace sdfg::gpu
