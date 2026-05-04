#pragma once

#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::analysis {

class BaseUserAnalyzer {
public:
    constexpr static int LOC_LOOP_READ_INIT = 0;
    constexpr static int LOC_LOOP_READ_CONDITION = 1;
    constexpr static int LOC_LOOP_READ_UPDATE = 2;

    enum class SymbolReadLocation {
        Assignment, // user will be the Transition, loc_index the sequence-idx
        LoopHeader, // user will be the StructuredLoop
        IfHeader, // user will be the if, loc_index the branch-idx
        MemletSubset, // user will be the memlet, loc_index the dimension-idx
        LibraryNode, // user will be libNode
    };

    enum class SymbolWriteLocation {
        Assignment, // user will be the transition
        LoopHeader // user will be the StructuredLoop. indvar is written multiple times (init, update) with this one
                   // call!
    };

    virtual ~BaseUserAnalyzer() = default;

    virtual void use_as_symbol_read(
        const std::string& container,
        const ControlFlowNode* node,
        const Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) = 0;
    virtual void use_as_symbol_write(
        const symbolic::Symbol& container, const ControlFlowNode* node, const Element* user, SymbolWriteLocation loc
    ) = 0;
    virtual void use_as_src_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) = 0;
    virtual void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) = 0;
    virtual void use_as_return_src(const std::string& container, const Return& ret) = 0;
};

/**
 * Finds uses of containers
 */
class BaseUserVisitor : public visitor::ActualStructuredSDFGVisitor, public virtual BaseUserAnalyzer {
public:
    virtual void handle_lib_node(Block&, data_flow::LibraryNode& libnode);

    virtual void handle_structured_loop_before_body(StructuredLoop& loop);
    virtual void handle_structured_loop_after_body(StructuredLoop& loop);

    bool visit(sdfg::structured_control_flow::Block& node) override;
    bool visit(sdfg::structured_control_flow::Sequence& node) override;
    bool visit(sdfg::structured_control_flow::IfElse& node) override;

    bool handleStructuredLoop(StructuredLoop& loop) override;

    bool visit(sdfg::structured_control_flow::Return& node) override;
};


} // namespace sdfg::analysis
