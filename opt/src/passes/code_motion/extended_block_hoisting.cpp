#include "sdfg/passes/code_motion/extended_block_hoisting.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/passes/offloading/code_motion/block_hoisting.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/offloading/data_offloading_node.h"

namespace sdfg {
namespace passes {

ExtendedBlockHoisting::
    ExtendedBlockHoisting(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
    : BlockHoisting(builder, analysis_manager) {}

bool ExtendedBlockHoisting::is_libnode_allowed(
    structured_control_flow::Sequence& body, data_flow::DataFlowGraph& dfg, data_flow::LibraryNode* libnode
) {
    if (BlockHoisting::is_libnode_allowed(body, dfg, libnode)) {
        return true;
    } else {
        return false;
    }
}

bool ExtendedBlockHoisting::equal_libnodes(structured_control_flow::Block& block1, structured_control_flow::Block& block2) {
    if (BlockHoisting::equal_libnodes(block1, block2)) {
        return true;
    }

    return false;
}

void ExtendedBlockHoisting::if_else_extract_invariant_libnode_front(
    structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
) {
    auto& first_block = static_cast<structured_control_flow::Block&>(if_else.at(0).first.at(0).first);
    auto& first_dfg = first_block.dataflow();
    auto* first_libnode = *first_dfg.library_nodes().begin();
    if (auto* offloading_node = dynamic_cast<offloading::DataOffloadingNode*>(first_libnode)) {
        if (offloading_node->is_d2h()) {
            auto* first_iedge = this->get_offloading_node_iedge(first_dfg, offloading_node);
            auto& first_src = static_cast<data_flow::AccessNode&>(first_iedge->src());
            std::string first_device_container = first_src.data();

            for (size_t i = 1; i < if_else.size(); i++) {
                auto& other_block = static_cast<structured_control_flow::Block&>(if_else.at(i).first.at(0).first);
                auto& other_dfg = other_block.dataflow();
                auto* other_offloading_node =
                    dynamic_cast<offloading::DataOffloadingNode*>(*other_dfg.library_nodes().begin());
                auto* other_iedge = this->get_offloading_node_iedge(other_dfg, other_offloading_node);
                auto& other_src = static_cast<data_flow::AccessNode&>(other_iedge->src());
                std::string other_device_container = other_src.data();

                if_else.at(i)
                    .first.replace(symbolic::symbol(other_device_container), symbolic::symbol(first_device_container));
            }
        } else if (offloading_node->is_h2d() || offloading_node->is_alloc()) {
            auto& first_oedge = *first_dfg.out_edges(*first_libnode).begin();
            auto& first_dst = static_cast<data_flow::AccessNode&>(first_oedge.dst());
            std::string first_device_container = first_dst.data();

            for (size_t i = 1; i < if_else.size(); i++) {
                auto& other_block = static_cast<structured_control_flow::Block&>(if_else.at(i).first.at(0).first);
                auto& other_dfg = other_block.dataflow();
                auto* other_libnode = *other_dfg.library_nodes().begin();
                auto& other_oedge = *other_dfg.out_edges(*other_libnode).begin();
                auto& other_dst = static_cast<data_flow::AccessNode&>(other_oedge.dst());
                std::string other_device_container = other_dst.data();

                if_else.at(i)
                    .first.replace(symbolic::symbol(other_device_container), symbolic::symbol(first_device_container));
            }
        }
    }


    BlockHoisting::if_else_extract_invariant_libnode_front(parent, if_else);
}

void ExtendedBlockHoisting::if_else_extract_invariant_libnode_back(
    structured_control_flow::Sequence& parent, structured_control_flow::IfElse& if_else
) {
    size_t first_size = if_else.at(0).first.size();
    auto& first_block = static_cast<structured_control_flow::Block&>(if_else.at(0).first.at(first_size - 1).first);
    auto& first_dfg = first_block.dataflow();
    auto* first_libnode = *first_dfg.library_nodes().begin();
    if (auto* offloading_node = dynamic_cast<offloading::DataOffloadingNode*>(first_libnode)) {
        if (offloading_node->is_d2h()) {
            auto* first_iedge = this->get_offloading_node_iedge(first_dfg, offloading_node);
            auto& first_src = static_cast<data_flow::AccessNode&>(first_iedge->src());
            std::string first_device_container = first_src.data();

            for (size_t i = 1; i < if_else.size(); i++) {
                size_t other_size = if_else.at(i).first.size();
                auto& other_block =
                    static_cast<structured_control_flow::Block&>(if_else.at(i).first.at(other_size - 1).first);
                auto& other_dfg = other_block.dataflow();
                auto* other_offloading_node =
                    dynamic_cast<offloading::DataOffloadingNode*>(*other_dfg.library_nodes().begin());
                auto* other_iedge = this->get_offloading_node_iedge(other_dfg, other_offloading_node);
                auto& other_src = static_cast<data_flow::AccessNode&>(other_iedge->src());
                std::string other_device_container = other_src.data();

                if_else.at(i)
                    .first.replace(symbolic::symbol(other_device_container), symbolic::symbol(first_device_container));
            }
        } else if (offloading_node->is_h2d()) {
            auto& first_oedge = *first_dfg.out_edges(*first_libnode).begin();
            auto& first_dst = static_cast<data_flow::AccessNode&>(first_oedge.dst());
            std::string first_device_container = first_dst.data();

            for (size_t i = 1; i < if_else.size(); i++) {
                size_t other_size = if_else.at(i).first.size();
                auto& other_block =
                    static_cast<structured_control_flow::Block&>(if_else.at(i).first.at(other_size - 1).first);
                auto& other_dfg = other_block.dataflow();
                auto* other_libnode = *other_dfg.library_nodes().begin();
                auto& other_oedge = *other_dfg.out_edges(*other_libnode).begin();
                auto& other_dst = static_cast<data_flow::AccessNode&>(other_oedge.dst());
                std::string other_device_container = other_dst.data();

                if_else.at(i)
                    .first.replace(symbolic::symbol(other_device_container), symbolic::symbol(first_device_container));
            }
        }
    }

    BlockHoisting::if_else_extract_invariant_libnode_back(parent, if_else);
}

bool ExtendedBlockHoisting::equal_offloading_nodes(
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
    if (dfg1.in_degree(*offloading_node1) != dfg1.in_degree(*offloading_node1)) {
        return false;
    }
    if (dfg2.in_degree(*offloading_node2) != dfg2.in_degree(*offloading_node2)) {
        return false;
    }

    // In edges:
    if (offloading_node1->has_transfer() || !offloading_node1->is_alloc()) {
        auto* iedge1 = this->get_offloading_node_iedge(dfg1, offloading_node1);
        auto* iedge2 = this->get_offloading_node_iedge(dfg2, offloading_node2);
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

        // Compare containers
        if (offloading_node1->is_h2d() || (!offloading_node1->has_transfer() && offloading_node1->is_free())) {
            auto& src1 = static_cast<data_flow::AccessNode&>(iedge1->src());
            auto& src2 = static_cast<data_flow::AccessNode&>(iedge2->src());
            if (src1.data() != src2.data()) {
                return false;
            }
        }
    }

    // Out edges:
    auto& oedge1 = *dfg1.out_edges(*offloading_node1).begin();
    auto& oedge2 = *dfg2.out_edges(*offloading_node2).begin();

    // Compare types
    if (oedge1.type() != oedge2.type()) {
        return false;
    }

    // Compare subsets
    if (oedge1.subset().size() != oedge2.subset().size()) {
        return false;
    }
    for (size_t i = 0; i < oedge1.subset().size(); i++) {
        if (!symbolic::eq(oedge1.subset().at(i), oedge2.subset().at(i))) {
            return false;
        }
    }

    // Compare containers
    if (offloading_node1->is_d2h() || (!offloading_node1->has_transfer() && offloading_node1->is_free())) {
        auto& dst1 = static_cast<data_flow::AccessNode&>(oedge1.dst());
        auto& dst2 = static_cast<data_flow::AccessNode&>(oedge2.dst());
        if (dst1.data() != dst2.data()) {
            return false;
        }
    }

    return true;
}

data_flow::Memlet* ExtendedBlockHoisting::
    get_offloading_node_iedge(data_flow::DataFlowGraph& dfg, offloading::DataOffloadingNode* offloading_node) {
    if (auto* external_offloading_node = dynamic_cast<offloading::ExternalDataOffloadingNode*>(offloading_node)) {
        for (auto& iedge : dfg.in_edges(*external_offloading_node)) {
            if (iedge.dst_conn() == external_offloading_node->input(external_offloading_node->transfer_index())) {
                return &iedge;
            }
        }
    } else {
        return &*dfg.in_edges(*offloading_node).begin();
    }
    return nullptr;
}

} // namespace passes
} // namespace sdfg
