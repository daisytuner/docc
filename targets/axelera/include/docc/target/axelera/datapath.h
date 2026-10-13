#pragma once

#include <optional>
#include <string>
#include <vector>

#include <sdfg/analysis/analysis.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/structured_control_flow/sequence.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/symbolic/symbolic.h>

namespace sdfg {
namespace axelera {

/// Operations of a DPU register program; each statement applies one.
enum class DpuOp { Copy, Add, Sub, Mul, Max, Min, Fma, Neg, Recip, Sqrt, Sin, Cos, Log2, Exp2 };

/// Elements of a pointer container read or written at consecutive indices across the lanes.
struct DatapathStream {
    std::string container;
    /// Element index as a function of the lane variable.
    symbolic::Expression index;
};

/// DPU register assignment: the next value popped from an input stream, a constant in every lane, or an operation on
/// registers.
struct DpuInstruction {
    size_t destination;
    std::optional<size_t> input_stream;
    std::optional<double> constant;
    DpuOp op = DpuOp::Copy;
    std::vector<size_t> operands;
};

/**
 * Datapath of a loop body run once per lane: input streams feed the IFDs and the DPU runs the program. A map body
 * stores register `result` through the ODR to `output`; a reduction body instead folds it into its accumulator
 * with `combine`.
 */
struct DatapathPlan {
    std::vector<DatapathStream> inputs;
    std::optional<DatapathStream> output;
    std::optional<DpuOp> combine;
    std::vector<DpuInstruction> program;
    size_t result;
    size_t num_registers;
};

/**
 * Maps the body of a loop over @p lane onto a datapath. With an @p accumulator, the body must end by combining a
 * value into that scalar instead of storing to memory.
 * @throws std::runtime_error naming the first construct a datapath cannot express
 */
DatapathPlan plan_datapath(
    const StructuredSDFG& sdfg,
    structured_control_flow::Sequence& body,
    const symbolic::Symbol& lane,
    const std::string& accumulator = ""
);

/// A Reduce loop whose accumulator is set to the reduction's identity right before it and stored right after it.
struct ReductionKernel {
    DpuOp combine;
    std::string accumulator;
    /// Positions of the initialization and the store in the loop's parent sequence.
    size_t init;
    size_t writeback;
    DatapathStream output;
};

/// @throws std::runtime_error naming why @p loop does not have that shape
ReductionKernel match_reduction(
    const StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager, structured_control_flow::Reduce& loop
);

} // namespace axelera
} // namespace sdfg
