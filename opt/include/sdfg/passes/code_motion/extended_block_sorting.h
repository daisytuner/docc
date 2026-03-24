#pragma once

#include <string>
#include <utility>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/passes/offloading/code_motion/block_sorting.h"
#include "sdfg/structured_control_flow/block.h"

namespace sdfg {
namespace passes {

class ExtendedBlockSortingPass : public BlockSortingPass {
protected:
    virtual bool is_libnode_side_effect_white_listed(data_flow::LibraryNode* libnode) override;

    virtual bool can_be_bubbled_up(structured_control_flow::Block& block) override;
    virtual bool can_be_bubbled_down(structured_control_flow::Block& block) override;

    virtual std::pair<int, std::string> get_prio_and_order(structured_control_flow::Block* block) override;

public:
    virtual std::string name() override { return "ExtendedBlockSorting"; }
};

} // namespace passes
} // namespace sdfg
