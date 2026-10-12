#pragma once

#include <optional>
#include <string>
#include <vector>

#include <sdfg/data_flow/tasklet.h>
#include <sdfg/structured_control_flow/sequence.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/symbolic/symbolic.h>

namespace sdfg {
namespace axelera {

/// Elements of a pointer container read or written at consecutive indices across the lanes.
struct DatapathStream {
    std::string container;
    /// Element index as a function of the lane variable.
    symbolic::Expression index;
};

/// DPU register assignment: either the next value popped from an input stream or an operation on registers.
struct DpuInstruction {
    size_t destination;
    std::optional<size_t> input_stream;
    data_flow::TaskletCode code = data_flow::TaskletCode::assign;
    std::vector<size_t> operands;
};

/// Elementwise datapath: input streams feed the IFDs, the DPU runs the program, the ODR stores one output stream.
struct DatapathPlan {
    std::vector<DatapathStream> inputs;
    DatapathStream output;
    std::vector<DpuInstruction> program;
    size_t result;
    size_t num_registers;
};

/**
 * Maps the body of a loop over @p lane, run once per lane, onto a datapath.
 * @throws std::runtime_error naming the first construct a datapath cannot express
 */
DatapathPlan
plan_datapath(const StructuredSDFG& sdfg, structured_control_flow::Sequence& body, const symbolic::Symbol& lane);

} // namespace axelera
} // namespace sdfg
