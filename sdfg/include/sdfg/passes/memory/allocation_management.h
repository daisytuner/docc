#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

class AllocationManagement : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    bool can_be_applied_allocation(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node);

    void apply_allocation(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node);

    bool can_be_applied_deallocation(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node);

    void apply_deallocation(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node);

public:
    AllocationManagement(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() {
        return "AllocationManagement";
    };

    bool accept(structured_control_flow::Block& node) override;
};

typedef VisitorPass<AllocationManagement> AllocationManagementPass;

} // namespace passes
} // namespace sdfg
