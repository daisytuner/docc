#include "sdfg/passes/code_motion/extended_block_sorting.h"

#include <string>
#include <utility>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/passes/offloading/code_motion/block_sorting.h"
#include "sdfg/structured_control_flow/block.h"

namespace sdfg {
namespace passes {

bool ExtendedBlockSortingPass::is_libnode_side_effect_white_listed(data_flow::LibraryNode* libnode) {
    return BlockSortingPass::is_libnode_side_effect_white_listed(libnode);
}

bool ExtendedBlockSortingPass::can_be_bubbled_up(structured_control_flow::Block& block) {
    return BlockSortingPass::can_be_bubbled_up(block);
}

bool ExtendedBlockSortingPass::can_be_bubbled_down(structured_control_flow::Block& block) {
    return BlockSortingPass::can_be_bubbled_down(block);
}

std::pair<int, std::string> ExtendedBlockSortingPass::get_prio_and_order(structured_control_flow::Block* block) {
    auto& dfg = block->dataflow();
    if (this->is_libnode_block(*block)) {
        auto* libnode = *dfg.library_nodes().begin();
        if (auto* external_offloading_node = dynamic_cast<offloading::ExternalDataOffloadingNode*>(libnode)) {
            std::string order = "";
            if (external_offloading_node->is_h2d() && external_offloading_node->is_alloc()) {
                for (auto& iedge : dfg.in_edges(*external_offloading_node)) {
                    if (iedge.dst_conn() ==
                        external_offloading_node->input(external_offloading_node->transfer_index())) {
                        auto& src = static_cast<data_flow::AccessNode&>(iedge.src());
                        order = src.data();
                        break;
                    }
                }
            } else if (external_offloading_node->is_d2h() && external_offloading_node->is_free()) {
                for (auto& oedge : dfg.out_edges(*external_offloading_node)) {
                    if (oedge.src_conn() ==
                        external_offloading_node->input(external_offloading_node->transfer_index())) {
                        auto& dst = dynamic_cast<data_flow::AccessNode&>(oedge.dst());
                        order = dst.data();
                        break;
                    }
                }
            }
            return {400, order};
        }
    }

    return BlockSortingPass::get_prio_and_order(block);
}

} // namespace passes
} // namespace sdfg
