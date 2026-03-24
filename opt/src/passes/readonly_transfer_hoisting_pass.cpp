#include "sdfg/passes/readonly_transfer_hoisting_pass.h"

#include <algorithm>
#include <cstddef>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/exceptions.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/tenstorrent/tenstorrent_offloading_node.h"

namespace sdfg {
namespace passes {

std::string ReadonlyTransferHoistingPass::name() { return "ReadonlyTransferHoistingPass"; }

bool ReadonlyTransferHoistingPass::
    run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    DEBUG_PRINTLN("Running ReadonlyTransferHoistingPass on " << builder.subject().name());
    bool applied = false;

    for (auto& container : builder.subject().containers()) {
        auto [sequence, index] = this->get_first_location(builder, analysis_manager, container);
        if (!sequence) {
            continue;
        }

        bool local_applied;
        std::set<size_t> visisted;
        do {
            local_applied =
                this->move_readonly_transfer(builder, analysis_manager, container, sequence, index, visisted);
            applied |= local_applied;
        } while (local_applied);
    }

    return applied;
}

std::pair<structured_control_flow::Sequence*, size_t> ReadonlyTransferHoistingPass::get_first_location(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, const std::string& container
) {
    auto& users = analysis_manager.get<analysis::Users>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    // Skip containers with views
    if (users.views(container).size() > 0) {
        return {nullptr, 0};
    }

    // The first location is the SDFG root for argument containers that are never written
    auto writes = users.writes(container);
    auto moves = users.moves(container);
    if (builder.subject().is_argument(container) && writes.size() == 0 && moves.size() == 0) {
        return {&builder.subject().root(), 0};
    }

    // Now, consider the case that the container is written/moved once
    analysis::User* user;
    if (writes.size() == 1 && moves.size() == 0) {
        user = writes.front();
    } else if (writes.size() == 0 && moves.size() == 1) {
        user = moves.front();
    } else {
        return {nullptr, 0};
    }

    auto* access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
    if (!access_node) {
        return {nullptr, 0};
    }

    auto& dfg = access_node->get_parent();
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    if (!block) {
        return {nullptr, 0};
    }

    auto* block_parent = scope_analysis.parent_scope(block);
    if (!block_parent) {
        return {nullptr, 0};
    }

    auto* block_parent_sequence = dynamic_cast<structured_control_flow::Sequence*>(block_parent);
    if (!block_parent_sequence) {
        return {nullptr, 0};
    }

    // All reads must be after the write/move
    size_t uses_after = 0;
    for (auto* other_user : users.all_uses_after(*user)) {
        if (other_user->container() == container) {
            uses_after++;
        }
    }
    if (uses_after != users.reads(container).size()) {
        return {nullptr, 0};
    }

    int index = block_parent_sequence->index(*block);
    if (index < 0 || index > block_parent_sequence->size()) {
        return {nullptr, 0};
    }

    if (index < block_parent_sequence->size() - 1) {
        index++;
    }
    return {block_parent_sequence, index};
}

bool ReadonlyTransferHoistingPass::move_readonly_transfer(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const std::string& container,
    structured_control_flow::Sequence* sequence,
    size_t index,
    std::set<size_t>& visisted
) {
    auto& users = analysis_manager.get<analysis::Users>();
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    // Check that every read belongs to an offloading node
    auto reads = users.reads(container);
    structured_control_flow::Sequence* source = nullptr;
    structured_control_flow::Block* source_block = nullptr;
    long long free_block_id = -1;
    for (auto* user : reads) {
        auto* access_node = dynamic_cast<data_flow::AccessNode*>(user->element());
        if (!access_node) {
            return false;
        }

        auto& dfg = access_node->get_parent();
        if (dfg.tasklets().size() != 0 || dfg.library_nodes().size() != 1) {
            return false;
        }

        auto* libnode = *dfg.library_nodes().begin();
        auto* offloading_node = dynamic_cast<offloading::DataOffloadingNode*>(libnode);
        if (!offloading_node) {
            return false;
        }

        // Skip external offloading nodes were the container is just an unused input
        if (auto* external_offloading_node = dynamic_cast<offloading::ExternalDataOffloadingNode*>(offloading_node)) {
            bool skip = true;
            for (auto& oedge : dfg.out_edges(*access_node)) {
                if (oedge.dst_conn() == external_offloading_node->input(external_offloading_node->transfer_index())) {
                    skip = false;
                    break;
                }
            }
            if (skip) {
                continue;
            }
        }

        if (offloading_node->is_d2h()) {
            return false;
        }

        auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
        if (!block) {
            return false;
        }

        if (visisted.contains(block->element_id())) {
            continue;
        }

        auto* block_parent = scope_analysis.parent_scope(block);
        if (!block_parent) {
            return false;
        }

        auto* block_parent_sequence = dynamic_cast<structured_control_flow::Sequence*>(block_parent);
        if (!block_parent_sequence) {
            return false;
        }

        if (source) {
            continue;
        }

        free_block_id = this->find_matching_free_block(block_parent_sequence, block, offloading_node);
        if (free_block_id < 0) {
            continue;
        }
        if (visisted.contains(free_block_id)) {
            return false;
        }

        source = block_parent_sequence;
        source_block = block;
    }
    if (!source || !source_block) {
        return false;
    }

    auto* offloading_node =
        static_cast<offloading::DataOffloadingNode*>(*source_block->dataflow().library_nodes().begin());
    auto [safe_sequence, safe_index] =
        this->get_safe_location(builder, analysis_manager, sequence, index, source_block, offloading_node);
    if (!safe_sequence) {
        return false;
    }

    size_t block_id = source_block->element_id();
    if (source == safe_sequence && source->index(*source_block) == safe_index) {
        visisted.insert(block_id);
        visisted.insert(free_block_id);
        return true;
    }

    builder.move_child(*source, source->index(*source_block), *safe_sequence, safe_index);
    visisted.insert(block_id);

    auto* free_block = dynamic_cast<structured_control_flow::Block*>(builder.find_element_by_id(free_block_id));
    if (!free_block) {
        throw InvalidSDFGException(
            "ReadonlyTransferHoistingPass: Could not find block with id: " + std::to_string(free_block_id)
        );
    }

    auto* free_block_parent = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(free_block));
    if (!free_block_parent) {
        throw InvalidSDFGException(
            "ReadonlyTransferHoistingPass: Cannot get the parent sequence of block with matching free"
        );
    }

    if (free_block_parent != safe_sequence || free_block_parent->index(*free_block) < safe_sequence->size()) {
        builder.move_child(*free_block_parent, free_block_parent->index(*free_block), *safe_sequence);
        visisted.insert(free_block_id);
    }
    analysis_manager.invalidate_all();
    DEBUG_PRINTLN("  Hoisted readonly transfer of container " << container);
    return true;
}

long long ReadonlyTransferHoistingPass::find_matching_free_block(
    structured_control_flow::Sequence* parent,
    structured_control_flow::Block* block,
    offloading::DataOffloadingNode* offloading_node
) {
    auto& oedge = *block->dataflow().out_edges(*offloading_node).begin();
    std::string device_container = static_cast<data_flow::AccessNode&>(oedge.dst()).data();

    for (size_t i = parent->index(*block) + 1; i < parent->size(); i++) {
        auto* other_block = dynamic_cast<structured_control_flow::Block*>(&parent->at(i).first);
        if (!other_block) {
            continue;
        }

        auto& dfg = other_block->dataflow();
        if (dfg.tasklets().size() != 0 || dfg.library_nodes().size() != 1) {
            continue;
        }

        auto* libnode = *dfg.library_nodes().begin();
        auto* other_offloading_node = dynamic_cast<offloading::DataOffloadingNode*>(libnode);
        if (!other_offloading_node) {
            continue;
        }

        if (other_offloading_node->has_transfer() || !other_offloading_node->is_free()) {
            continue;
        }

        auto& other_oedge = *other_block->dataflow().out_edges(*other_offloading_node).begin();
        std::string other_device_container = static_cast<data_flow::AccessNode&>(other_oedge.dst()).data();
        if (other_device_container != device_container) {
            continue;
        }

        if (dynamic_cast<cuda::CUDADataOffloadingNode*>(offloading_node) &&
            dynamic_cast<cuda::CUDADataOffloadingNode*>(other_offloading_node)) {
            return other_block->element_id();
        }
        if (dynamic_cast<tenstorrent::TTDataOffloadingNode*>(offloading_node) &&
            dynamic_cast<tenstorrent::TTDataOffloadingNode*>(other_offloading_node)) {
            return other_block->element_id();
        }
        if (dynamic_cast<offloading::ExternalDataOffloadingNode*>(offloading_node) &&
            dynamic_cast<offloading::ExternalDataOffloadingNode*>(other_offloading_node)) {
            return other_block->element_id();
        }
    }
    return -1;
}

std::pair<structured_control_flow::Sequence*, size_t> ReadonlyTransferHoistingPass::get_safe_location(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Sequence* sequence,
    size_t index,
    structured_control_flow::Block* block,
    offloading::DataOffloadingNode* offloading_node
) {
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto parents = this->get_parents(scope_analysis, block);
    auto [result_sequence, result_index] = this->correct_location(scope_analysis, sequence, index, parents);
    if (!result_sequence) {
        return {nullptr, 0};
    }

    std::vector<std::string> containers;
    for (auto& symbol : offloading_node->symbols()) {
        containers.push_back(symbol->get_name());
    }
    if (auto* external_offloading_node = dynamic_cast<offloading::ExternalDataOffloadingNode*>(offloading_node)) {
        for (auto& iedge : block->dataflow().in_edges(*external_offloading_node)) {
            if (iedge.dst_conn() == external_offloading_node->input(external_offloading_node->transfer_index()) ||
                iedge.dst_conn() == "_ret") {
                continue;
            }
            auto& src = static_cast<data_flow::AccessNode&>(iedge.src());
            if (dynamic_cast<data_flow::ConstantNode*>(&src)) {
                continue;
            }
            containers.push_back(src.data());
        }
    }

    if (containers.size() == 0) {
        return {result_sequence, result_index};
    }

    for (auto& container : containers) {
        auto [new_sequence, new_index] = this->get_first_location(builder, analysis_manager, container);
        if (!new_sequence) {
            return {nullptr, 0};
        }

        auto [corrected_sequence, corrected_index] =
            this->correct_location(scope_analysis, new_sequence, new_index, parents);
        if (!corrected_sequence) {
            return {nullptr, 0};
        }

        bool found = false;
        for (auto [parent_sequence, parent_index] : parents) {
            if (corrected_sequence == parent_sequence && result_sequence == parent_sequence) {
                if (corrected_index <= parent_index && result_index <= parent_index) {
                    result_index = std::max(corrected_index, result_index);
                    found = true;
                    break;
                } else {
                    return {nullptr, 0};
                }
            } else if (corrected_sequence == parent_sequence) {
                if (corrected_index <= parent_index) {
                    result_sequence = corrected_sequence;
                    result_index = corrected_index;
                    found = true;
                    break;
                } else {
                    return {nullptr, 0};
                }
            } else if (result_sequence == parent_sequence) {
                if (result_index <= parent_index) {
                    found = true;
                    break;
                } else {
                    return {nullptr, 0};
                }
            }
        }
        if (!found) {
            return {nullptr, 0};
        }
    }

    return {result_sequence, result_index};
}

std::pair<structured_control_flow::Sequence*, size_t> ReadonlyTransferHoistingPass::correct_location(
    analysis::ScopeAnalysis& scope_analysis,
    structured_control_flow::Sequence* sequence,
    size_t index,
    const std::vector<std::pair<structured_control_flow::Sequence*, size_t>>& parents
) {
    for (auto [parent_sequence, parent_index] : parents) {
        if (sequence == parent_sequence) {
            if (index <= parent_index) {
                return {sequence, index};
            } else {
                return {nullptr, 0};
            }
        }
    }

    structured_control_flow::ControlFlowNode* current = &sequence->at(index).first;
    structured_control_flow::ControlFlowNode* current_parent = sequence;
    while (current_parent != nullptr) {
        if (auto* current_sequence = dynamic_cast<structured_control_flow::Sequence*>(current_parent)) {
            for (auto [parent_sequence, parent_index] : parents) {
                if (current_sequence == parent_sequence) {
                    size_t current_index = current_sequence->index(*current);
                    if (current_index < parent_index) {
                        return {current_sequence, current_index + 1};
                    } else {
                        return {nullptr, 0};
                    }
                }
            }
        }
        current = current_parent;
        current_parent = scope_analysis.parent_scope(current_parent);
    }

    return {nullptr, 0};
}

std::vector<std::pair<structured_control_flow::Sequence*, size_t>> ReadonlyTransferHoistingPass::
    get_parents(analysis::ScopeAnalysis& scope_analysis, structured_control_flow::Block* block) {
    std::vector<std::pair<structured_control_flow::Sequence*, size_t>> parents;
    structured_control_flow::ControlFlowNode* current = block;
    structured_control_flow::ControlFlowNode* current_parent = scope_analysis.parent_scope(block);
    while (current_parent != nullptr) {
        if (auto* current_sequence = dynamic_cast<structured_control_flow::Sequence*>(current_parent)) {
            parents.push_back({current_sequence, current_sequence->index(*current)});
        }
        current = current_parent;
        current_parent = scope_analysis.parent_scope(current_parent);
    }
    return parents;
}

} // namespace passes
} // namespace sdfg
