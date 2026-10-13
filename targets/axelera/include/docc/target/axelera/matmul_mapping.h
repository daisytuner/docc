#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>

namespace sdfg {
namespace axelera {

/**
 * Runs every float matmul at the top level of the SDFG on the MVM with a fixed recipe instead of an agent: GEMM
 * nodes expand into loop nests, @ref MvmEinsumTransform quantizes them and splits the rows of A over the most AI
 * cores up to the configured number that divide them, the row loop is tiled, row tiles of A and of the products and
 * the weights are staged in L1, the weights stay resident in the IMC, and row tiles are double-buffered.
 * @return true if a matmul was mapped
 */
bool map_matmuls(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

} // namespace axelera
} // namespace sdfg
