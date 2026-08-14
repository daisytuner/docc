#pragma once

#include <unordered_map>
#include <unordered_set>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg::analysis {

class NewLoopDependencyAnalysis;

class DepAccess {
    friend class NewLoopDependencyAnalysis;

    std::string container_;
    ControlFlowNode* scope_;
    Element* element_;
    data_flow::Subset subset_;

public:
    DepAccess(const std::string& container, ControlFlowNode* scope, Element* element, const data_flow::Subset& subset)
        : container_(container), scope_(scope), element_(element), subset_(subset) {}

    bool is_indirect();
};

class DataDependencyAnalyzer : public virtual BaseUserAnalyzer {
    const StructuredSDFG& sdfg_;
    std::vector<std::unique_ptr<DepAccess>> accesses_;

protected:
    DepAccess& create_symbol_read(
        const std::string& container,
        ControlFlowNode* node,
        Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    );
    DepAccess& create_symbol_write(
        const symbolic::Symbol& container, ControlFlowNode* node, Element* user, SymbolWriteLocation loc
    );

    DepAccess&
    create_direct_access(const std::string& container, ControlFlowNode* sdfg_node, data_flow::AccessNode* access_node);
    DepAccess& create_indirect_access(
        const std::string& container, Block& block, data_flow::AccessNode& access_node, data_flow::Memlet& edge
    );

    virtual void direct_read(DepAccess& access) = 0;
    virtual void direct_write(DepAccess& access) = 0;
    virtual void indirect_read(DepAccess& access) = 0;
    virtual void indirect_write(DepAccess& access) = 0;
    virtual void aliasing_source(const std::string& container) = 0;

public:
    DataDependencyAnalyzer(const StructuredSDFG& sdfg) : sdfg_(sdfg) {}

    void use_as_return_src(const std::string& container, Return& ret) override;

    void use_as_symbol_read(
        const std::string& container,
        ControlFlowNode* node,
        Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override;

    void use_as_src_node(const std::string& container, data_flow::AccessNode& node, data_flow::Memlet& edge, Block& block)
        override;

    void use_as_dst_node(const std::string& container, data_flow::AccessNode& node, data_flow::Memlet& edge, Block& block)
        override;

    void use_as_symbol_write(
        const symbolic::Symbol& container, ControlFlowNode* node, Element* user, SymbolWriteLocation loc
    ) override;
};

} // namespace sdfg::analysis
