#pragma once

#include <optional>
#include <vector>

#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/tiles/library_nodes/pipeline_node.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg::gpu {

/**
 * @brief Interleave the register-staged prefetch loads (and, double-buffered, the stores of the next panel) of a
 *        software-pipelined MMA panel loop with its matrix-core instructions.
 *
 * Targets the steady-state panel loop that SoftwarePipelining (register-staged, peeled) and TileVectorizer produce.
 * Its body is one panel step, or two step sequences when double-buffered. A step is branch-free: it issues the
 * prefetch loads into registers (@ref tiles::TileCopyNode in phase LoadRegs), runs a constant-trip k-step loop of LDS
 * fragment loads and MMA nodes, (double-buffered) stores the next panel from registers into the other buffer (phase
 * StoreRegs), and ends at a barrier. Without a schedule the loads are fenced in front of the compute, so their latency
 * is hidden only by other waves.
 *
 * apply() inserts a @ref tiles::PipelineScheduleNode before each step's barrier and marks the loads scheduled
 * (dropping their fence). Its recipe, over the unrolled step:
 *
 *     lds_read(r)                                              // fragments of the first k-step
 *     for each k-step k, each MMA m of the k-step:
 *         mfma(1)
 *         every `global_read_spacing`-th MMA: global_read(1)    // until all prefetch loads are placed
 *         otherwise, k >= `lds_write_start`: lds_write(1)        // until all stores are placed
 *         m == lds_read_position, k not last: lds_read(r)       // fragments of the next k-step
 *     remaining global reads and LDS writes
 *
 * with
 *  - global reads: the per-thread transfer count of the loads, from their plans, widths and thread counts; the
 *    spacing shrinks to `MMAs / loads` when the loads would not fit otherwise;
 *  - LDS writes: the per-thread transfer count of the stores, each split into row-aligned pieces on padded buffers;
 *  - MMAs per k-step: the MMA nodes in the k-step loop body; k-steps: its constant trip count;
 *  - r: @p lds_reads_per_step, or derived as the LDS fragment bytes each lane reads per k-step divided by 16 (the
 *    widest LDS read per lane), from the fragment shapes and the enclosing wave map's lane count.
 *
 * All steps must yield the same recipe. A recipe that does not match the emitted instructions exactly only weakens
 * the schedule; semantics are unchanged.
 */
class GpuMmaInterleave : public transformations::Transformation {
    structured_control_flow::StructuredLoop& loop_;
    size_t global_read_spacing_;
    size_t lds_read_position_;
    size_t lds_reads_per_step_;
    size_t lds_write_start_;

public:
    GpuMmaInterleave(
        structured_control_flow::StructuredLoop& loop,
        size_t global_read_spacing = 3,
        size_t lds_read_position = 1,
        size_t lds_reads_per_step = 0,
        size_t lds_write_start = 1
    );

    std::string name() const override {
        return "GpuMmaInterleave";
    }

    /// The schedule apply() would insert, or nullopt if the transformation does not apply.
    std::optional<std::vector<tiles::ScheduleGroup>> recipe(builder::StructuredSDFGBuilder& builder) const;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static GpuMmaInterleave from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace sdfg::gpu
