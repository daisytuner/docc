#pragma once

#include <string>

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/visualizer/visualizer.h"

namespace sdfg {
namespace visualizer {

class DotVisualizer : public Visualizer {
private:
    std::string last_comp_name_;
    std::string last_comp_name_cluster_;
    bool show_block_ids = false;

    virtual void visualizeBlock(const StructuredSDFG& sdfg, const structured_control_flow::Block& block) override;
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

public:
    using Visualizer::Visualizer;

    virtual void visualize() override;

    static void writeToFile(const StructuredSDFG& sdfg, const std::filesystem::path& file);
    static void writeToFile(const StructuredSDFG& sdfg, const std::filesystem::path* file = nullptr);
};

} // namespace visualizer
} // namespace sdfg
