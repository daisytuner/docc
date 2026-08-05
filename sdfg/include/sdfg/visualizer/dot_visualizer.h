#pragma once

#include <string>

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/visualizer/visualizer.h"

namespace sdfg {
namespace visualizer {

class DotVisualizer : public Visualizer {
private:
    struct SeqChainElem {
        const Element* element;
        std::string node_id;
        std::string cluster_id;
    };
    struct SeqChainScope {
        std::optional<SeqChainElem> last_chain_elem;
        std::optional<SeqChainElem> last2_chain_elem;
    };
    std::list<SeqChainScope> seq_scope_stack_{SeqChainScope()};
    bool show_block_ids = true;

    void register_chain_elem(const Element& element, const std::string& node_id, const std::string& cluster_id);
    void enter_scope();
    void exit_scope();

    virtual void visualizeSDFG(const SDFG& sdfg) override;

    virtual void visualizeStructuredSDFG(const StructuredSDFG& sdfg) override;
    virtual void visualizeBlock(const StructuredSDFG& sdfg, const structured_control_flow::Block& block) override;
    virtual void visualizeAssignmentBlock(
        const StructuredSDFG& sdfg, const structured_control_flow::AssignmentBlock& assignment_block
    ) override;
    virtual void visualizeSequence(const StructuredSDFG& sdfg, const structured_control_flow::Sequence& sequence)
        override;
    virtual void visualizeIfElse(const StructuredSDFG& sdfg, const structured_control_flow::IfElse& if_else) override;
    virtual void visualizeWhile(const StructuredSDFG& sdfg, const structured_control_flow::While& while_loop) override;
    virtual void visualizeFor(const StructuredSDFG& sdfg, const structured_control_flow::For& loop) override;
    virtual void visualizeReturn(const StructuredSDFG& sdfg, const structured_control_flow::Return& return_node)
        override;
    virtual void visualizeBreak(const StructuredSDFG& sdfg, const structured_control_flow::Break& break_node) override;
    virtual void visualizeContinue(const StructuredSDFG& sdfg, const structured_control_flow::Continue& continue_node)
        override;
    virtual void visualizeMap(const StructuredSDFG& sdfg, const structured_control_flow::Map& map_node) override;
    virtual void visualizeReduce(const StructuredSDFG& sdfg, const structured_control_flow::Reduce& reduce_node)
        override;

    virtual void visualizeDataFlowGraph(const std::string& id, const data_flow::DataFlowGraph& dfg) override;

public:
    using Visualizer::Visualizer;

    static void writeToFile(const Function& sdfg, const std::filesystem::path& file);
    static void writeToFile(const Function& sdfg, const std::filesystem::path* file = nullptr);
};

} // namespace visualizer
} // namespace sdfg
