#include "sdfg/analysis/data_dependency_analyzer.h"

#include <cassert>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <isl/ctx.h>
#include <isl/options.h>
#include <isl/set.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"
#include "sdfg/symbolic/polyhedral.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"

namespace sdfg::analysis {

bool DepAccess::is_indirect() { return !subset_.empty(); }

DepAccess& DataDependencyAnalyzer::create_symbol_read(
    const std::string& container,
    ControlFlowNode* node,
    Element* user,
    SymbolReadLocation loc,
    int loc_index,
    symbolic::Expression expr
) {
    // which expression does the read is no longer identifiable...
    auto ptr = std::make_unique<DepAccess>(container, node, user, data_flow::Subset{});
    accesses_.push_back(std::move(ptr));
    return *accesses_.back().get();
}

DepAccess& DataDependencyAnalyzer::create_symbol_write(
    const symbolic::Symbol& container, ControlFlowNode* node, Element* user, SymbolWriteLocation loc
) {
    // exact write operation not identifiable anymore...
    auto ptr = std::make_unique<DepAccess>(container->get_name(), node, user, data_flow::Subset{});
    accesses_.push_back(std::move(ptr));
    return *accesses_.back().get();
}

DepAccess& DataDependencyAnalyzer::
    create_direct_access(const std::string& container, ControlFlowNode* node, data_flow::AccessNode* access_node) {
    Element* elem = (access_node != nullptr) ? access_node : static_cast<Element*>(node);
    auto ptr = std::make_unique<DepAccess>(container, node, elem, data_flow::Subset{});
    accesses_.push_back(std::move(ptr));
    return *accesses_.back().get();
}

DepAccess& DataDependencyAnalyzer::create_indirect_access(
    const std::string& container, Block& block, data_flow::AccessNode& node, data_flow::Memlet& edge
) {
    auto ptr = std::make_unique<DepAccess>(container, &block, &node, edge.subset());
    accesses_.push_back(std::move(ptr));
    return *accesses_.back().get();
}

void DataDependencyAnalyzer::use_as_return_src(const std::string& container, Return& ret) {
    auto& access = create_direct_access(container, &ret, nullptr);
    direct_read(access);
}

void DataDependencyAnalyzer::use_as_symbol_read(
    const std::string& container,
    ControlFlowNode* node,
    Element* user,
    SymbolReadLocation loc,
    int loc_index,
    symbolic::Expression expr
) {
    auto& access = create_symbol_read(container, node, user, loc, loc_index, expr);
    direct_read(access);
}

void DataDependencyAnalyzer::
    use_as_src_node(const std::string& container, data_flow::AccessNode& node, data_flow::Memlet& edge, Block& block) {
    if (dyn_cast<data_flow::ConstantNode>(&node)) {
        return;
    }

    if (edge.is_src_read()) {
        auto& dir_access = create_direct_access(container, &block, &node);
        direct_read(dir_access);
        if (edge.is_src_pointed_to_read()) {
            auto& indir_access = create_indirect_access(container, block, node, edge);
            indirect_read(indir_access);
        }
    } else {
        aliasing_source(container);
    }
}

void DataDependencyAnalyzer::
    use_as_dst_node(const std::string& container, data_flow::AccessNode& node, data_flow::Memlet& edge, Block& block) {
    if (edge.is_dst_write()) {
        auto& dir_access = create_direct_access(container, &block, &node);
        direct_write(dir_access);
    } else if (edge.is_dst_pointed_to_write()) {
        auto& dir_access = create_direct_access(container, &block, &node);
        direct_read(dir_access);
        auto& indir_access = create_indirect_access(container, block, node, edge);
        indirect_write(indir_access);
    } else {
        throw std::runtime_error("Unexpected data flow edge -> #" + std::to_string(node.element_id()));
    }
}

void DataDependencyAnalyzer::use_as_symbol_write(
    const symbolic::Symbol& container, ControlFlowNode* node, Element* user, SymbolWriteLocation loc
) {
    auto& access = create_symbol_write(container, node, user, loc);
    direct_write(access);
}

} // namespace sdfg::analysis
