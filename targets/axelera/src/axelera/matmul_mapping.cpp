#include "docc/target/axelera/matmul_mapping.h"

#include <algorithm>
#include <optional>
#include <vector>

#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/passes/expansion/library_node_expansion_pass.h>
#include <sdfg/passes/structured_control_flow/sequence_fusion.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/tiles/transformations/local_storage.h>
#include <sdfg/tiles/transformations/software_pipelining.h>
#include <sdfg/transformations/loop_tiling.h>
#include <sdfg/visitor/for_each.h>

#include "docc/target/axelera/axelera_scheduler.h"
#include "docc/target/axelera/matmul_node.h"
#include "docc/target/axelera/matmul_transforms.h"

namespace sdfg {
namespace axelera {

namespace {

/// Rows of A per L1 tile.
constexpr int64_t TILE_ROWS = 32;

/// Expands the float GEMMs at the top level into the zero-initialization and accumulation nests the MVM transform
/// matches.
bool expand_gemms(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& root = builder.subject().root();
    bool expanded = false;
    // An expansion replaces its block, so the scan restarts after each.
    for (bool again = true; again;) {
        again = false;
        for (size_t i = 0; i < root.size() && !again; ++i) {
            auto block = dynamic_cast<structured_control_flow::Block*>(&root.at(i));
            if (!block) {
                continue;
            }
            for (auto* node : block->dataflow().library_nodes()) {
                auto gemm = dynamic_cast<math::blas::GEMMNode*>(node);
                if (gemm && gemm->precision() == math::blas::BLAS_Precision::s) {
                    again = passes::expansion::expand_single_math_node(builder, *block, *gemm).expanded;
                    break;
                }
            }
        }
        expanded |= again;
    }
    if (expanded) {
        passes::SequenceFusion().run(builder, analysis_manager);
    }
    return expanded;
}

/// The access feeding @p connector of the single MVM node under @p scope.
const data_flow::AccessNode& mvm_operand(structured_control_flow::ControlFlowNode& scope, const std::string& connector) {
    const data_flow::AccessNode* found = nullptr;
    visitor::for_each_block(scope, [&](structured_control_flow::Block& block) {
        for (auto* node : block.dataflow().library_nodes()) {
            if (!dynamic_cast<AxeleraMatMulNode*>(node)) {
                continue;
            }
            for (auto& memlet : block.dataflow().in_edges(*node)) {
                if (memlet.dst_conn() == connector) {
                    found = &static_cast<const data_flow::AccessNode&>(memlet.src());
                }
            }
        }
    });
    if (!found) {
        throw std::runtime_error("Axelera matmul mapping: no MVM operand " + connector);
    }
    return *found;
}

void apply(
    transformations::Transformation& transformation,
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager
) {
    if (!transformation.can_be_applied(builder, analysis_manager)) {
        throw transformations::
            InvalidTransformationException("Axelera matmul mapping: " + transformation.name() + " does not apply");
    }
    transformation.apply(builder, analysis_manager);
}

/// Largest divisor of @p n not above @p limit.
int64_t largest_divisor(int64_t n, int64_t limit) {
    for (auto divisor = std::min(n, limit); divisor > 1; --divisor) {
        if (n % divisor == 0) {
            return divisor;
        }
    }
    return 1;
}

/// The most AI cores up to @p cores that the MVM transform can split the matmul at @p loop over.
std::optional<int64_t> mvm_cores(
    structured_control_flow::StructuredLoop& loop,
    int64_t cores,
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager
) {
    for (auto count = cores; count >= 1; --count) {
        if (MvmEinsumTransform(loop, 0, count).can_be_applied(builder, analysis_manager)) {
            return count;
        }
    }
    return std::nullopt;
}

void map_matmul(
    structured_control_flow::StructuredLoop& loop,
    int64_t cores,
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager
) {
    MvmEinsumTransform mvm(loop, 0, cores);
    apply(mvm, builder, analysis_manager);
    auto& rows = *mvm.row_loop();
    auto count = SymEngine::down_cast<const SymEngine::Integer&>(*rows.num_iterations()).as_int();

    auto tile_rows = largest_divisor(count, TILE_ROWS);
    transformations::LoopTiling tiling(rows, tile_rows);
    apply(tiling, builder, analysis_manager);
    auto& tiles = *tiling.outer_loop();

    for (auto connector : {"A", "Y"}) {
        transformations::LocalStorage stage(rows, mvm_operand(rows, connector));
        apply(stage, builder, analysis_manager);
    }
    transformations::LocalStorage weights(tiles, mvm_operand(tiles, "B"));
    apply(weights, builder, analysis_manager);
    ImcResidency residency(tiles);
    apply(residency, builder, analysis_manager);

    if (count > tile_rows) {
        transformations::SoftwarePipelining pipelining(tiles, 2);
        apply(pipelining, builder, analysis_manager);
    }
}

} // namespace

bool map_matmuls(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    expand_gemms(builder, analysis_manager);
    auto cores = analysis_manager.options().get(NUM_CORES, int64_t{1});

    // A mapping removes the zero-initialization nest before its matmul, which the scan has passed already.
    auto& root = builder.subject().root();
    std::vector<structured_control_flow::StructuredLoop*> loops;
    for (size_t i = 0; i < root.size(); ++i) {
        if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(i))) {
            loops.push_back(loop);
        }
    }
    bool mapped = false;
    for (auto* loop : loops) {
        if (auto count = mvm_cores(*loop, cores, builder, analysis_manager)) {
            map_matmul(*loop, *count, builder, analysis_manager);
            mapped = true;
        }
    }
    return mapped;
}

} // namespace axelera
} // namespace sdfg
