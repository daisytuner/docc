#pragma once

#include <string>
#include <utility>
#include <vector>

#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/sdfg.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace visualizer {

class Visualizer {
protected:
    codegen::PrettyPrinter stream_;
    const Function& sdfg_;
    std::vector<std::pair<const std::string, const std::string>> replacements_;

    virtual std::string expression(const std::string expr);

    // SDFG nodes
    virtual void visualizeSDFG(const SDFG& sdfg) = 0;

    // Structured SDFG nodes
    virtual void visualizeStructuredSDFG(const StructuredSDFG& sdfg) = 0;
    virtual void visualizeNode(const StructuredSDFG& sdfg, const structured_control_flow::ControlFlowNode& node);
    virtual void visualizeBlock(const StructuredSDFG& sdfg, const structured_control_flow::Block& block) = 0;
    virtual void visualizeSequence(const StructuredSDFG& sdfg, const structured_control_flow::Sequence& sequence) = 0;
    virtual void visualizeIfElse(const StructuredSDFG& sdfg, const structured_control_flow::IfElse& if_else) = 0;
    virtual void visualizeWhile(const StructuredSDFG& sdfg, const structured_control_flow::While& while_loop) = 0;
    virtual void visualizeFor(const StructuredSDFG& sdfg, const structured_control_flow::For& loop) = 0;
    virtual void visualizeReturn(const StructuredSDFG& sdfg, const structured_control_flow::Return& return_node) = 0;
    virtual void visualizeBreak(const StructuredSDFG& sdfg, const structured_control_flow::Break& break_node) = 0;
    virtual void visualizeContinue(const StructuredSDFG& sdfg, const structured_control_flow::Continue& continue_node) = 0;
    virtual void visualizeMap(const StructuredSDFG& sdfg, const structured_control_flow::Map& map_node) = 0;
    virtual void visualizeReduce(const StructuredSDFG& sdfg, const structured_control_flow::Reduce& reduce_node) = 0;

    // Data Flow Graph
    virtual void visualizeDataFlowGraph(const std::string& id, const data_flow::DataFlowGraph& dfg) = 0;
    virtual void visualizeTasklet(data_flow::Tasklet const& tasklet);
    virtual void visualizeForBounds(
        symbolic::Symbol const& indvar,
        symbolic::Expression const& init,
        symbolic::Condition const& condition,
        symbolic::Expression const& update
    );

    std::string subsetRangeString(data_flow::Subset const& subset, int subIdx);

public:
    Visualizer(const Function& sdfg) : stream_{}, sdfg_{sdfg}, replacements_{} {};

    virtual void visualize();

    codegen::PrettyPrinter const& getStream() const { return this->stream_; }

    virtual void visualizeSubset(data_flow::Subset const& begin_sub, types::IType const* type = nullptr, int subIdx = 0);
};

} // namespace visualizer
} // namespace sdfg
