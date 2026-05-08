#include "sdfg/passes/memory/allocation_management.h"

#include "sdfg/analysis/dominance_analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"

#include "sdfg/data_flow/library_nodes/stdlib/stdlib.h"

namespace sdfg {
namespace passes {

AllocationManagement::
    AllocationManagement(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool AllocationManagement::can_be_applied_allocation(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node) {
    symbolic::Expression allocation_size = SymEngine::null;
    const data_flow::LibraryNode* libnode = nullptr;
    if (library_node.code() == stdlib::LibraryNodeType_Alloca) {
        if (auto alloca_node = dynamic_cast<stdlib::AllocaNode*>(&library_node)) {
            libnode = &library_node;
            allocation_size = alloca_node->size();
        }
    } else if (library_node.code() == stdlib::LibraryNodeType_Malloc) {
        if (auto malloc_node = dynamic_cast<stdlib::MallocNode*>(&library_node)) {
            libnode = &library_node;
            allocation_size = malloc_node->size();
        }
    }
    if (libnode == nullptr || allocation_size.is_null()) {
        return false;
    }

    // Retrieve allocated container
    auto& sdfg = this->builder_.subject();
    auto& oedge = *graph.out_edges(library_node).begin();
    auto& dst = static_cast<data_flow::AccessNode&>(oedge.dst());
    const std::string& container = dst.data();
    auto& type = sdfg.type(container);
    if (type.type_id() != types::TypeID::Pointer) {
        return false;
    }

    // Limitations
    if (graph.out_degree(dst) != 0) {
        return false;
    }
    if (graph.in_degree(dst) != 1) {
        return false;
    }
    if (graph.in_degree(*libnode) != 0) {
        return false;
    }

    // Limitations
    auto& block = static_cast<structured_control_flow::Block&>(*graph.get_parent());
    auto& scope_analysis = this->analysis_manager_.get<analysis::ScopeAnalysis>();
    if (scope_analysis.parent_scope(&block) != &sdfg.root()) {
        return false;
    }

    // Criterion 1: Allocation size only depends on parameters
    for (auto& sym : symbolic::atoms(allocation_size)) {
        if (!sdfg.is_argument(sym->get_name())) {
            return false;
        }
    }

    // Criterion 2: Allocation dominates all uses
    auto& users_analysis = this->analysis_manager_.get<analysis::Users>();
    auto& dominance_analysis = this->analysis_manager_.get<analysis::DominanceAnalysis>();
    auto uses = users_analysis.uses(container);
    analysis::User* allocation_user = users_analysis.get_user(container, &dst, analysis::Use::WRITE);
    for (auto& use : uses) {
        if (use == allocation_user) {
            continue;
        }
        if (!dominance_analysis.dominates(*allocation_user, *use)) {
            return false;
        }
    }

    return true;
}

void AllocationManagement::apply_allocation(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node) {
    symbolic::Expression allocation_size = SymEngine::null;
    std::string storage_type_val;
    if (library_node.code() == stdlib::LibraryNodeType_Alloca) {
        if (auto alloca_node = dynamic_cast<stdlib::AllocaNode*>(&library_node)) {
            allocation_size = alloca_node->size();
            storage_type_val = "CPU_Stack";
        }
    } else if (library_node.code() == stdlib::LibraryNodeType_Malloc) {
        if (auto malloc_node = dynamic_cast<stdlib::MallocNode*>(&library_node)) {
            allocation_size = malloc_node->size();
            storage_type_val = "CPU_Heap";
        }
    }

    auto& sdfg = this->builder_.subject();
    auto& oedge = *graph.out_edges(library_node).begin();
    auto& dst = static_cast<data_flow::AccessNode&>(oedge.dst());
    const std::string& container = dst.data();

    // Update storage type
    auto& type = sdfg.type(container);

    auto new_type = type.clone();
    types::StorageType new_storage_type = types::StorageType(
        storage_type_val,
        allocation_size,
        types::StorageType::AllocationType::Managed,
        type.storage_type().deallocation()
    );
    new_type->storage_type(new_storage_type);

    builder_.change_type(container, *new_type);

    // Remove allocation node
    auto& block = static_cast<structured_control_flow::Block&>(*graph.get_parent());
    builder_.clear_access_node_legacy(block, dst);
}

bool AllocationManagement::
    can_be_applied_deallocation(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node) {
    const data_flow::LibraryNode* libnode = nullptr;
    if (library_node.code() == stdlib::LibraryNodeType_Free) {
        if (auto free_node = dynamic_cast<stdlib::FreeNode*>(&library_node)) {
            libnode = &library_node;
        }
    }
    if (libnode == nullptr) {
        return false;
    }

    // Retrieve allocated container
    auto& sdfg = this->builder_.subject();
    auto& iedge = *graph.in_edge_for_connector(library_node, library_node.input(0));
    auto& src = static_cast<const data_flow::AccessNode&>(iedge.src());
    const std::string& container = src.data();
    auto& type = sdfg.type(container);
    if (type.type_id() != types::TypeID::Pointer) {
        return false;
    }

    // Limitations
    auto& block = static_cast<structured_control_flow::Block&>(*graph.get_parent());
    auto& scope_analysis = this->analysis_manager_.get<analysis::ScopeAnalysis>();
    if (scope_analysis.parent_scope(&block) != &sdfg.root()) {
        return false;
    }

    // Criterion 2: Allocation post-dominates all uses
    auto& users_analysis = this->analysis_manager_.get<analysis::Users>();
    auto& dominance_analysis = this->analysis_manager_.get<analysis::DominanceAnalysis>();
    auto uses = users_analysis.uses(container);
    analysis::User* deallocation_user =
        users_analysis.get_user(container, const_cast<data_flow::AccessNode*>(&src), analysis::Use::READ);
    for (auto& use : uses) {
        if (use == deallocation_user) {
            continue;
        }
        if (!dominance_analysis.post_dominates(*deallocation_user, *use)) {
            return false;
        }
    }

    return true;
}

void AllocationManagement::apply_deallocation(data_flow::DataFlowGraph& graph, data_flow::LibraryNode& library_node) {
    auto& sdfg = this->builder_.subject();
    auto& iedge = *graph.in_edge_for_connector(library_node, library_node.input(0));
    auto& src = static_cast<const data_flow::AccessNode&>(iedge.src());
    const std::string& container = src.data();

    // Update storage type
    auto& type = sdfg.type(container);

    auto new_type = type.clone();
    types::StorageType new_storage_type = types::StorageType(
        "CPU_Heap",
        type.storage_type().allocation_size(),
        type.storage_type().allocation(),
        types::StorageType::AllocationType::Managed
    );
    new_type->storage_type(new_storage_type);

    builder_.change_type(container, *new_type);

    // Remove allocation node
    auto& block = static_cast<structured_control_flow::Block&>(*graph.get_parent());
    builder_.clear_node(block, library_node);
}

bool AllocationManagement::accept(structured_control_flow::Block& node) {
    bool applied = false;

    auto& graph = node.dataflow();
    for (auto& lib_node : graph.library_nodes()) {
        if (can_be_applied_allocation(graph, *lib_node)) {
            apply_allocation(graph, *lib_node);
            applied = true;
            continue;
        }
        if (can_be_applied_deallocation(graph, *lib_node)) {
            apply_deallocation(graph, *lib_node);
            applied = true;
            continue;
        }
    }
    return applied; // Return whether any node was modified
}

} // namespace passes
} // namespace sdfg
