#include "sdfg/passes/offloading/code_motion/block_hoisting.h"
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/alloca.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"
#include "sdfg/data_flow/library_nodes/stdlib/memmove.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

BlockHoisting::BlockHoisting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {};

bool BlockHoisting::accept(structured_control_flow::Map& map_stmt) {
    auto& parent = static_cast<structured_control_flow::Sequence&>(*map_stmt.get_parent());

    bool applied = false;
    applied |= this->map_invariant_front(parent, map_stmt);
    applied |= this->map_invariant_back(parent, map_stmt);

    return applied;
}

bool BlockHoisting::accept(structured_control_flow::IfElse& if_else) {
    // Ignore incomplete branches for now
    if (if_else.size() == 0 || !if_else.is_complete()) {
        return false;
    }

    auto& parent = static_cast<structured_control_flow::Sequence&>(*if_else.get_parent());

    bool applied = false;
    applied |= this->if_else_invariant_front(parent, if_else);
    applied |= this->if_else_invariant_back(parent, if_else);

    return applied;
}

bool BlockHoisting::is_libnode_allowed(
    structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg, data_flow::LibraryNode* libnode
) {
    if (dynamic_cast<stdlib::AllocaNode*>(libnode)) {
        return true;
    } else if (dynamic_cast<stdlib::MemcpyNode*>(libnode)) {
        return true;
    } else if (dynamic_cast<stdlib::MemmoveNode*>(libnode)) {
        return true;
    } else if (dynamic_cast<stdlib::MemsetNode*>(libnode)) {
        return true;
    } else if (dynamic_cast<offloading::DataOffloadingNode*>(libnode)) {
        return true;
    } else {
        return false;
    }
}

bool BlockHoisting::equal_libnodes(structured_control_flow::Block& block1, structured_control_flow::Block& block2) {
    if (equal_lib_blocks(block1, block2)) {
        return true;
    }

    auto* libnode1 = *block1.dataflow().library_nodes().begin();
    auto* libnode2 = *block2.dataflow().library_nodes().begin();
    if (auto* offloading_node1 = dynamic_cast<offloading::DataOffloadingNode*>(libnode1)) {
        if (auto* offloading_node2 = dynamic_cast<offloading::DataOffloadingNode*>(libnode2)) {
            return this->equal_offloading_nodes(block1, offloading_node1, block2, offloading_node2);
        }
    }
    return false;
}

bool BlockHoisting::equal_lib_blocks(structured_control_flow::Block& block1, structured_control_flow::Block& block2) {
    auto& dfg1 = block1.dataflow();
    auto& dfg2 = block2.dataflow();

    // Get library nodes
    auto* libnode1 = *dfg1.library_nodes().begin();
    auto* libnode2 = *dfg2.library_nodes().begin();

    // Collect in edges
    std::unordered_map<std::string, data_flow::Memlet*> iedges1, iedges2;
    for (auto& iedge : dfg1.in_edges(*libnode1)) {
        iedges1.insert({iedge.dst_conn(), &iedge});
    }
    for (auto& iedge : dfg2.in_edges(*libnode2)) {
        iedges2.insert({iedge.dst_conn(), &iedge});
    }

    // In edges must have the same type, subset, and container
    if (iedges1.size() != iedges2.size()) {
        return false;
    }
    for (auto [conn, iedge1] : iedges1) {
        if (!iedges2.contains(conn)) {
            return false;
        }
        auto* iedge2 = iedges2.at(conn);

        // Compare types
        if (iedge1->type() != iedge2->type()) {
            return false;
        }

        // Compare subsets
        if (iedge1->subset().size() != iedge2->subset().size()) {
            return false;
        }
        for (size_t i = 0; i < iedge1->subset().size(); i++) {
            if (!symbolic::eq(iedge1->subset().at(i), iedge2->subset().at(i))) {
                return false;
            }
        }

        // Compare containers
        auto& src1 = static_cast<data_flow::AccessNode&>(iedge1->src());
        auto& src2 = static_cast<data_flow::AccessNode&>(iedge2->src());
        if (!data_flow::AccessNode::identicalBackingData(src1, src2)) {
            return false;
        }
    }

    // Collect out edges
    std::unordered_map<std::string, data_flow::Memlet*> oedges1, oedges2;
    for (auto& oedge : dfg1.out_edges(*libnode1)) {
        oedges1.insert({oedge.src_conn(), &oedge});
    }
    for (auto& oedge : dfg2.out_edges(*libnode2)) {
        oedges2.insert({oedge.src_conn(), &oedge});
    }

    // Out edges must have the same type, subset, and container
    if (oedges1.size() != oedges2.size()) {
        return false;
    }
    for (auto [conn, oedge1] : oedges1) {
        if (!oedges2.contains(conn)) {
            return false;
        }
        auto& oedge2 = oedges2.at(conn);

        // Compare types
        if (oedge1->type() != oedge2->type()) {
            return false;
        }

        // Compare subsets
        if (oedge1->subset().size() != oedge2->subset().size()) {
            return false;
        }
        for (size_t i = 0; i < oedge1->subset().size(); i++) {
            if (!symbolic::eq(oedge1->subset().at(i), oedge2->subset().at(i))) {
                return false;
            }
        }

        // Compare containers
        auto& dst1 = static_cast<data_flow::AccessNode&>(oedge1->dst());
        auto& dst2 = static_cast<data_flow::AccessNode&>(oedge2->dst());
        if (!data_flow::AccessNode::identicalBackingData(dst1, dst2)) {
            return false;
        }
    }

    // Checks library node internals
    return this->equal_libnodes(libnode1, libnode2);
}

void BlockHoisting::if_else_extract_invariant_libnode_front(
    structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
) {
    auto& first_block = static_cast<structured_control_flow::Block&>(if_else.at(0).first.at(0).first);
    auto& first_dfg = first_block.dataflow();
    if (!first_dfg.library_nodes().empty()) {
        auto* first_libnode = *first_dfg.library_nodes().begin();
        if (auto* offloading_node = dynamic_cast<offloading::DataOffloadingNode*>(first_libnode)) {
            auto dev_in_idx = offloading_node->dev_ptr_input_idx();
            auto dev_out_idx = offloading_node->dev_ptr_output_idx();
            if (dev_in_idx >= 0) {
                auto* first_iedge =
                    first_dfg.in_edge_for_connector(*offloading_node, offloading_node->inputs().at(dev_in_idx));
                auto& first_src = static_cast<const data_flow::AccessNode&>(first_iedge->src());
                std::string first_device_container = first_src.data();

                for (size_t i = 1; i < if_else.size(); i++) {
                    auto& other_block = static_cast<structured_control_flow::Block&>(if_else.at(i).first.at(0).first);
                    auto& other_dfg = other_block.dataflow();
                    auto* other_offloading_node =
                        dynamic_cast<offloading::DataOffloadingNode*>(*other_dfg.library_nodes().begin());
                    auto* other_iedge =
                        other_dfg.in_edge_for_connector(*other_offloading_node, other_offloading_node->dev_in_conn());
                    auto& other_src = static_cast<const data_flow::AccessNode&>(other_iedge->src());
                    std::string other_device_container = other_src.data();

                    if_else.at(i)
                        .first
                        .replace(symbolic::symbol(other_device_container), symbolic::symbol(first_device_container));
                }
            } else if (dev_out_idx >= 0) {
                auto& first_oedge =
                    **first_dfg.out_edges_for_connector(*first_libnode, first_libnode->outputs().at(dev_out_idx))
                          .begin();
                auto& first_dst = static_cast<const data_flow::AccessNode&>(first_oedge.dst());
                std::string first_device_container = first_dst.data();

                for (size_t i = 1; i < if_else.size(); i++) {
                    auto& other_block = static_cast<structured_control_flow::Block&>(if_else.at(i).first.at(0).first);
                    auto& other_dfg = other_block.dataflow();
                    auto* other_libnode =
                        dynamic_cast<offloading::DataOffloadingNode*>(*other_dfg.library_nodes().begin());
                    auto& other_oedge =
                        **other_dfg.out_edges_for_connector(*other_libnode, other_libnode->dev_out_conn()).begin();
                    auto& other_dst = static_cast<const data_flow::AccessNode&>(other_oedge.dst());
                    std::string other_device_container = other_dst.data();

                    if_else.at(i)
                        .first
                        .replace(symbolic::symbol(other_device_container), symbolic::symbol(first_device_container));
                }
            }
        }
    }
    // This function is a wrapper that can be overridden in sub-classes
    this->if_else_extract_invariant_front(parent, if_else);
}

void BlockHoisting::if_else_extract_invariant_libnode_back(
    structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
) {
    size_t first_size = if_else.at(0).first.size();
    auto& first_block = static_cast<structured_control_flow::Block&>(if_else.at(0).first.at(first_size - 1).first);
    auto& first_dfg = first_block.dataflow();
    if (!first_dfg.library_nodes().empty()) {
        auto* first_libnode = *first_dfg.library_nodes().begin();
        if (auto* offloading_node = dynamic_cast<offloading::DataOffloadingNode*>(first_libnode)) {
            auto dev_in_idx = offloading_node->dev_ptr_input_idx();
            auto dev_out_idx = offloading_node->dev_ptr_output_idx();
            if (dev_in_idx >= 0) {
                auto* first_iedge =
                    &*first_dfg.in_edge_for_connector(*offloading_node, offloading_node->inputs().at(dev_in_idx));
                auto& first_src = static_cast<const data_flow::AccessNode&>(first_iedge->src());
                std::string first_device_container = first_src.data();

                for (size_t i = 1; i < if_else.size(); i++) {
                    size_t other_size = if_else.at(i).first.size();
                    auto& other_block =
                        static_cast<structured_control_flow::Block&>(if_else.at(i).first.at(other_size - 1).first);
                    auto& other_dfg = other_block.dataflow();
                    auto* other_offloading_node =
                        dynamic_cast<offloading::DataOffloadingNode*>(*other_dfg.library_nodes().begin());
                    auto* other_iedge =
                        other_dfg.in_edge_for_connector(*other_offloading_node, other_offloading_node->dev_in_conn());
                    auto& other_src = static_cast<const data_flow::AccessNode&>(other_iedge->src());
                    std::string other_device_container = other_src.data();

                    if_else.at(i)
                        .first
                        .replace(symbolic::symbol(other_device_container), symbolic::symbol(first_device_container));
                }
            } else if (dev_out_idx >= 0) {
                auto& first_oedge =
                    **first_dfg.out_edges_for_connector(*offloading_node, offloading_node->outputs().at(dev_out_idx))
                          .begin();
                auto& first_dst = static_cast<const data_flow::AccessNode&>(first_oedge.dst());
                std::string first_device_container = first_dst.data();

                for (size_t i = 1; i < if_else.size(); i++) {
                    size_t other_size = if_else.at(i).first.size();
                    auto& other_block =
                        static_cast<structured_control_flow::Block&>(if_else.at(i).first.at(other_size - 1).first);
                    auto& other_dfg = other_block.dataflow();
                    auto* other_libnode =
                        dynamic_cast<offloading::DataOffloadingNode*>(*other_dfg.library_nodes().begin());
                    auto& other_oedge =
                        **other_dfg.out_edges_for_connector(*other_libnode, other_libnode->dev_out_conn()).begin();
                    auto& other_dst = static_cast<const data_flow::AccessNode&>(other_oedge.dst());
                    std::string other_device_container = other_dst.data();

                    if_else.at(i)
                        .first
                        .replace(symbolic::symbol(other_device_container), symbolic::symbol(first_device_container));
                }
            }
        }
    }
    // This function is a wrapper that can be overridden in sub-classes
    this->if_else_extract_invariant_back(parent, if_else);
}

bool BlockHoisting::is_invariant_move(
    structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg, bool no_loop_carried_dependencies
) {
    if (dfg.nodes().size() != 2) {
        return false;
    }
    if (dfg.edges().size() != 1) {
        return false;
    }
    auto& edge = *dfg.edges().begin();
    if (edge.type() != data_flow::MemletType::Dereference_Src) {
        return false;
    }

    if (no_loop_carried_dependencies) {
        return true;
    }
    auto& src = static_cast<data_flow::AccessNode&>(edge.src());

    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_view(users_analysis, body);
    if (!body_view.writes(src.data()).empty() || !body_view.moves(src.data()).empty()) {
        return false;
    }

    return true;
}

bool BlockHoisting::is_invariant_view(
    structured_control_flow::Sequence& body,
    data_flow::DataFlowGraph& dfg,
    symbolic::Symbol indvar,
    bool no_loop_carried_dependencies
) {
    if (dfg.nodes().size() != 2) {
        return false;
    }
    if (dfg.edges().size() != 1) {
        return false;
    }
    auto& edge = *dfg.edges().begin();
    if (edge.type() != data_flow::MemletType::Reference) {
        return false;
    }

    if (!indvar.is_null()) {
        auto& subset = edge.subset();
        for (const auto& dim : subset) {
            if (symbolic::uses(dim, indvar)) {
                return false;
            }
        }
    }

    if (no_loop_carried_dependencies) {
        return true;
    }
    auto& src = static_cast<data_flow::AccessNode&>(edge.src());

    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_view(users_analysis, body);
    if (!body_view.writes(src.data()).empty() || !body_view.moves(src.data()).empty()) {
        return false;
    }

    return true;
}

bool BlockHoisting::is_invariant_libnode(
    structured_control_flow::Sequence& body,
    data_flow::DataFlowGraph& dfg,
    symbolic::Symbol indvar,
    bool no_loop_carried_dependencies
) {
    if (dfg.library_nodes().size() != 1) {
        return false;
    }
    if (dfg.tasklets().size() != 0) {
        return false;
    }
    auto* libnode = *dfg.library_nodes().begin();
    if (!libnode) {
        return false;
    }
    if (dfg.data_nodes().size() != libnode->outputs().size() + libnode->inputs().size()) {
        return false;
    }

    if (!this->is_libnode_allowed(body, dfg, libnode)) {
        return false;
    }

    if (!indvar.is_null()) {
        if (libnode->symbols().contains(indvar)) {
            return false;
        }
        for (auto& oedge : dfg.out_edges(*libnode)) {
            for (const auto& dim : oedge.subset()) {
                if (symbolic::uses(dim, indvar)) {
                    return false;
                }
            }
        }
        for (auto& iedge : dfg.in_edges(*libnode)) {
            for (const auto& dim : iedge.subset()) {
                if (symbolic::uses(dim, indvar)) {
                    return false;
                }
            }
        }
    }

    if (no_loop_carried_dependencies) {
        return true;
    }

    auto& users_analysis = analysis_manager_.get<analysis::Users>();
    analysis::UsersView body_view(users_analysis, body);
    for (auto& iedge : dfg.in_edges(*libnode)) {
        auto& src = static_cast<data_flow::AccessNode&>(iedge.src());
        if (!body_view.writes(src.data()).empty() || !body_view.moves(src.data()).empty()) {
            return false;
        }
    }
    for (auto& oedge : dfg.out_edges(*libnode)) {
        auto& dst = static_cast<data_flow::AccessNode&>(oedge.dst());
        if (!body_view.writes(dst.data()).empty() || !body_view.moves(dst.data()).empty()) {
            return false;
        }
    }

    return true;
}

bool BlockHoisting::equal_moves(structured_control_flow::Block& block1, structured_control_flow::Block& block2) {
    // Edges must have same type and subset
    auto& edge1 = *block1.dataflow().edges().begin();
    auto& edge2 = *block2.dataflow().edges().begin();
    if (edge1.type() != edge2.type()) {
        return false;
    }
    if (edge1.subset().size() != edge2.subset().size()) {
        return false;
    }
    for (size_t i = 0; i < edge1.subset().size(); i++) {
        if (!symbolic::eq(edge1.subset().at(i), edge2.subset().at(i))) {
            return false;
        }
    }

    // Directions must be the same
    if (edge1.src_conn() != edge2.src_conn()) {
        return false;
    }
    if (edge1.dst_conn() != edge2.dst_conn()) {
        return false;
    }

    // src's must have the same containers
    auto& src1 = static_cast<data_flow::AccessNode&>(edge1.src());
    auto& src2 = static_cast<data_flow::AccessNode&>(edge2.src());
    if (!data_flow::AccessNode::identicalBackingData(src1, src2)) {
        return false;
    }

    // dst's must have the same containers
    auto& dst1 = static_cast<data_flow::AccessNode&>(edge1.dst());
    auto& dst2 = static_cast<data_flow::AccessNode&>(edge2.dst());
    if (!data_flow::AccessNode::identicalBackingData(dst1, dst2)) {
        return false;
    }

    return true;
}

bool BlockHoisting::equal_views(structured_control_flow::Block& block1, structured_control_flow::Block& block2) {
    // Edges must have same type and subset
    auto& edge1 = *block1.dataflow().edges().begin();
    auto& edge2 = *block2.dataflow().edges().begin();
    if (edge1.type() != edge2.type()) {
        return false;
    }
    if (edge1.subset().size() != edge2.subset().size()) {
        return false;
    }
    for (size_t i = 0; i < edge1.subset().size(); i++) {
        if (!symbolic::eq(edge1.subset().at(i), edge2.subset().at(i))) {
            return false;
        }
    }

    // src's must have the same containers
    auto& src1 = static_cast<data_flow::AccessNode&>(edge1.src());
    auto& src2 = static_cast<data_flow::AccessNode&>(edge2.src());
    if (!data_flow::AccessNode::identicalBackingData(src1, src2)) {
        return false;
    }

    // dst's must have the same containers
    auto& dst1 = static_cast<data_flow::AccessNode&>(edge1.dst());
    auto& dst2 = static_cast<data_flow::AccessNode&>(edge2.dst());
    if (!data_flow::AccessNode::identicalBackingData(dst1, dst2)) {
        return false;
    }

    return true;
}

bool BlockHoisting::equal_libnodes(data_flow::LibraryNode* libnode1, data_flow::LibraryNode* libnode2) {
    if (auto* alloca_node1 = dynamic_cast<stdlib::AllocaNode*>(libnode1)) {
        if (auto* alloca_node2 = dynamic_cast<stdlib::AllocaNode*>(libnode2)) {
            return symbolic::eq(alloca_node1->size(), alloca_node2->size());
        }
    }
    if (auto* memcpy_node1 = dynamic_cast<stdlib::MemcpyNode*>(libnode1)) {
        if (auto* memcpy_node2 = dynamic_cast<stdlib::MemcpyNode*>(libnode2)) {
            return symbolic::eq(memcpy_node1->count(), memcpy_node2->count());
        }
    }
    if (auto* memmove_node1 = dynamic_cast<stdlib::MemmoveNode*>(libnode1)) {
        if (auto* memmove_node2 = dynamic_cast<stdlib::MemmoveNode*>(libnode2)) {
            return symbolic::eq(memmove_node1->count(), memmove_node2->count());
        }
    }
    if (auto* memset_node1 = dynamic_cast<stdlib::MemsetNode*>(libnode1)) {
        if (auto* memset_node2 = dynamic_cast<stdlib::MemsetNode*>(libnode2)) {
            return symbolic::eq(memset_node1->value(), memset_node2->value()) &&
                   symbolic::eq(memset_node1->num(), memset_node2->num());
        }
    }

    return false;
}

bool BlockHoisting::map_invariant_front(structured_control_flow::Sequence& parent, structured_control_flow::Map& map_stmt) {
    // Extract first child
    auto& body = map_stmt.root();
    if (body.size() == 0) {
        return false;
    }
    auto first_child = body.at(0);
    if (!first_child.second.assignments().empty()) {
        return false;
    }
    auto& first_node = first_child.first;

    auto* block = dynamic_cast<structured_control_flow::Block*>(&first_node);
    if (!block) {
        return false;
    }
    if (this->map_invariant_move(parent, map_stmt, *block)) {
        return true;
    }
    if (this->map_invariant_view(parent, map_stmt, *block)) {
        return true;
    }
    if (this->map_invariant_libnode_front(parent, map_stmt, *block)) {
        return true;
    }

    return false;
}

bool BlockHoisting::map_invariant_back(structured_control_flow::Sequence& parent, structured_control_flow::Map& map_stmt) {
    // Extract last child
    auto& body = map_stmt.root();
    if (body.size() == 0) {
        return false;
    }
    auto last_child = body.at(body.size() - 1);
    if (!last_child.second.assignments().empty()) {
        return false;
    }
    auto& last_node = last_child.first;

    auto* block = dynamic_cast<structured_control_flow::Block*>(&last_node);
    if (!block) {
        return false;
    }
    return this->map_invariant_libnode_back(parent, map_stmt, *block);
}

bool BlockHoisting::map_invariant_move(
    structured_control_flow::Sequence& parent,
    structured_control_flow::Map& map_stmt,
    structured_control_flow::Block& block
) {
    auto& body = map_stmt.root();
    if (!this->is_invariant_move(body, block.dataflow())) {
        return false;
    }

    size_t map_index = parent.index(map_stmt);
    builder_.move_child(body, 0, parent, map_index);
    return true;
}

bool BlockHoisting::map_invariant_view(
    structured_control_flow::Sequence& parent,
    structured_control_flow::Map& map_stmt,
    structured_control_flow::Block& block
) {
    auto& body = map_stmt.root();
    if (!this->is_invariant_view(body, block.dataflow(), map_stmt.indvar())) {
        return false;
    }

    size_t map_index = parent.index(map_stmt);
    builder_.move_child(body, 0, parent, map_index);
    return true;
}

bool BlockHoisting::map_invariant_libnode_front(
    structured_control_flow::Sequence& parent,
    structured_control_flow::Map& map_stmt,
    structured_control_flow::Block& block
) {
    // For now, only allow libnode hoisting on sequential maps
    if (map_stmt.schedule_type().value() != ScheduleType_Sequential::value()) {
        return false;
    }

    auto& body = map_stmt.root();
    if (!this->is_invariant_libnode(body, block.dataflow(), map_stmt.indvar())) {
        return false;
    }

    size_t map_index = parent.index(map_stmt);
    builder_.move_child(body, 0, parent, map_index);
    return true;
}

bool BlockHoisting::map_invariant_libnode_back(
    structured_control_flow::Sequence& parent,
    structured_control_flow::Map& map_stmt,
    structured_control_flow::Block& block
) {
    // For now, only allow libnode hoisting on sequential maps
    if (map_stmt.schedule_type().value() != ScheduleType_Sequential::value()) {
        return false;
    }

    auto& body = map_stmt.root();
    if (!this->is_invariant_libnode(body, block.dataflow(), map_stmt.indvar())) {
        return false;
    }

    size_t map_index = parent.index(map_stmt);
    builder_.move_child(body, body.size() - 1, parent, map_index + 1);
    return true;
}

bool BlockHoisting::
    if_else_invariant_front(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else) {
    // Extract the first block of first case
    if (if_else.at(0).first.size() == 0) {
        return false;
    }
    auto first_child = if_else.at(0).first.at(0);
    if (!first_child.second.assignments().empty()) {
        return false;
    }
    auto* first_block = dynamic_cast<structured_control_flow::Block*>(&first_child.first);
    if (!first_block) {
        return false;
    }

    // Collect the first block of all other cases
    std::vector<structured_control_flow::Block*> other_blocks;
    for (size_t i = 1; i < if_else.size(); i++) {
        if (if_else.at(i).first.size() == 0) {
            return false;
        }
        auto other_child = if_else.at(i).first.at(0);
        if (!other_child.second.assignments().empty()) {
            return false;
        }
        auto* other_block = dynamic_cast<structured_control_flow::Block*>(&other_child.first);
        if (!other_block) {
            return false;
        }
        other_blocks.push_back(other_block);
    }

    // Check the first block of all cases for invariant move / view / libnode
    if (this->is_invariant_move(if_else.at(0).first, first_block->dataflow())) {
        for (size_t i = 0; i < other_blocks.size(); i++) {
            if (!this->is_invariant_move(if_else.at(i + 1).first, other_blocks[i]->dataflow())) {
                return false;
            }
            if (!this->equal_moves(*first_block, *other_blocks[i])) {
                return false;
            }
        }
        this->if_else_extract_invariant_front(parent, if_else);
        return true;
    } else if (this->is_invariant_view(if_else.at(0).first, first_block->dataflow())) {
        for (size_t i = 0; i < other_blocks.size(); i++) {
            if (!this->is_invariant_view(if_else.at(i + 1).first, other_blocks[i]->dataflow())) {
                return false;
            }
            if (!this->equal_views(*first_block, *other_blocks[i])) {
                return false;
            }
        }
        this->if_else_extract_invariant_front(parent, if_else);
        return true;
    } else if (this->is_invariant_libnode(if_else.at(0).first, first_block->dataflow())) {
        for (size_t i = 0; i < other_blocks.size(); i++) {
            if (!this->is_invariant_libnode(if_else.at(i + 1).first, other_blocks[i]->dataflow())) {
                return false;
            }
            if (!this->equal_libnodes(*first_block, *other_blocks[i])) {
                return false;
            }
        }
        this->if_else_extract_invariant_libnode_front(parent, if_else);
        return true;
    }

    return false;
}

bool BlockHoisting::
    if_else_invariant_back(structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else) {
    // Extract the last block of first case
    if (if_else.at(0).first.size() == 0) {
        return false;
    }
    auto last_child = if_else.at(0).first.at(if_else.at(0).first.size() - 1);
    if (!last_child.second.assignments().empty()) {
        return false;
    }
    auto* last_block = dynamic_cast<structured_control_flow::Block*>(&last_child.first);
    if (!last_block) {
        return false;
    }

    // Collect the last block of all other cases
    std::vector<structured_control_flow::Block*> other_blocks;
    for (size_t i = 1; i < if_else.size(); i++) {
        if (if_else.at(i).first.size() == 0) {
            return false;
        }
        auto other_child = if_else.at(i).first.at(if_else.at(i).first.size() - 1);
        if (!other_child.second.assignments().empty()) {
            return false;
        }
        auto* other_block = dynamic_cast<structured_control_flow::Block*>(&other_child.first);
        if (!other_block) {
            return false;
        }
        other_blocks.push_back(other_block);
    }

    // Check the first block of all cases for invariant move / view / libnode
    if (this->is_invariant_libnode(if_else.at(0).first, last_block->dataflow())) {
        for (size_t i = 0; i < other_blocks.size(); i++) {
            if (!this->is_invariant_libnode(if_else.at(i + 1).first, other_blocks[i]->dataflow())) {
                return false;
            }
            if (!this->equal_libnodes(*last_block, *other_blocks[i])) {
                return false;
            }
        }
        this->if_else_extract_invariant_libnode_back(parent, if_else);
        return true;
    }

    return false;
}

void BlockHoisting::if_else_extract_invariant_front(
    structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
) {
    size_t if_else_index = parent.index(if_else);
    builder_.move_child(if_else.at(0).first, 0, parent, if_else_index);

    for (size_t i = 1; i < if_else.size(); i++) {
        builder_.remove_child(if_else.at(i).first, 0);
    }
}

void BlockHoisting::if_else_extract_invariant_back(
    structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
) {
    size_t if_else_index = parent.index(if_else);
    builder_.move_child(if_else.at(0).first, if_else.at(0).first.size() - 1, parent, if_else_index + 1);

    for (size_t i = 1; i < if_else.size(); i++) {
        builder_.remove_child(if_else.at(i).first, if_else.at(i).first.size() - 1);
    }
}

bool BlockHoisting::equal_offloading_nodes(
    structured_control_flow::Block& block1,
    offloading::DataOffloadingNode* offloading_node1,
    structured_control_flow::Block& block2,
    offloading::DataOffloadingNode* offloading_node2
) {
    if (!offloading_node1->equal_with(*offloading_node2)) {
        return false;
    }

    // Check in/out degree
    auto& dfg1 = block1.dataflow();
    auto& dfg2 = block2.dataflow();
    if (dfg1.in_degree(*offloading_node1) != dfg2.in_degree(*offloading_node2)) {
        return false;
    }
    if (dfg1.out_degree(*offloading_node1) != dfg2.out_degree(*offloading_node2)) {
        return false;
    }

    // In edges:
    for (int i = 0; i < offloading_node1->inputs().size(); i++) {
        auto* iedge1 = dfg1.in_edge_for_connector(*offloading_node1, offloading_node1->input(i));
        auto* iedge2 = dfg2.in_edge_for_connector(*offloading_node2, offloading_node2->input(i));
        if (!iedge1 || !iedge2) {
            return false;
        }

        // Compare types
        if (iedge1->type() != iedge2->type()) {
            return false;
        }

        // Compare subsets
        if (iedge1->subset().size() != iedge2->subset().size()) {
            return false;
        }
        for (size_t i = 0; i < iedge1->subset().size(); i++) {
            if (!symbolic::eq(iedge1->subset().at(i), iedge2->subset().at(i))) {
                return false;
            }
        }

        if (i == offloading_node1->host_ptr_input_idx()) { // only host side containers can be relied upon to match
            auto& src1 = static_cast<const data_flow::AccessNode&>(iedge1->src());
            auto& src2 = static_cast<const data_flow::AccessNode&>(iedge2->src());
            if (!data_flow::AccessNode::identicalBackingData(src1, src2)) {
                return false;
            }
        }
    }

    // Out edges:
    for (int i = 0; i < offloading_node1->outputs().size(); i++) {
        auto oedges1 = dfg1.out_edges_for_connector(*offloading_node1, offloading_node1->output(0));
        auto oedges2 = dfg2.out_edges_for_connector(*offloading_node2, offloading_node2->output(0));
        if (oedges1.size() != oedges2.size()) {
            return false;
        }
        for (int j = 0; j < oedges1.size(); j++) {
            auto oedge1 = oedges1.at(j);
            auto oedge2 = oedges2.at(j);

            if (oedge1 && oedge2) {
                // Compare types
                if (oedge1->type() != oedge2->type()) {
                    return false;
                }

                // Compare subsets
                if (oedge1->subset().size() != oedge2->subset().size()) {
                    return false;
                }
                for (size_t k = 0; k < oedge1->subset().size(); k++) {
                    if (!symbolic::eq(oedge1->subset().at(k), oedge2->subset().at(k))) {
                        return false;
                    }
                }

                // out edges can only be device side in new model, we only can rely on host containers being the same,
                // so do not check th
            } else if (!oedge1 ^ !oedge2) {
                return false;
            }
        }
    }

    return true;
}

} // namespace passes
} // namespace sdfg
